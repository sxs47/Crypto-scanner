"""
Local web dashboard for the pump scanner.

    python app.py            # then open http://127.0.0.1:5000

Candle CSVs are loaded into memory once; scans for a given threshold/window
are computed on demand and cached. The "Update data" button runs the same
incremental Bybit download as scanner.py in a background thread. The
Signals page serves the research report and live scores from signals.py.
The live loop fetches each newly closed 15m candle and raises setup alerts
(setups.py rules) for the Alerts page.
"""

import json
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import html
from datetime import timedelta

from flask import Flask, abort, jsonify, redirect, request, send_from_directory, session

import auth
import collector
import news
import notify
import blowoff
import carry
import prepump
import radar

import exchange
import futures
import journal
import live
import scanner
import setups
import signals
import trend

# Some symbols are non-ASCII; Windows consoles default to cp1252
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config.update(SECRET_KEY=auth.secret_key(), SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
                  PERMANENT_SESSION_LIFETIME=timedelta(days=30))


# --------------------------------------------------------------------------- #
# Login (only when a password has been set: python app.py --set-password)
# --------------------------------------------------------------------------- #
@app.before_request
def require_login():
    if not auth.enabled() or request.path in ("/login", "/logout"):
        return None
    if session.get("auth") == auth.session_token():
        return None
    if request.path.startswith("/api/"):
        return jsonify({"error": "login required"}), 401
    return redirect("/login")


def _login_page(error=""):
    nxt = request.values.get("next", "/")
    page = auth.LOGIN_PAGE.replace("{next}", html.escape(nxt)).replace("{error}", f'<div class="err">{html.escape(error)}</div>' if error else "")
    return page, 401 if error else 200


@app.route("/login", methods=["GET", "POST"])
def login():
    if not auth.enabled():
        return redirect("/")
    if request.method == "GET":
        return _login_page()
    ip = request.headers.get("X-Forwarded-For", request.remote_addr or "").split(",")[0].strip()
    wait = auth.locked_out(ip)
    if wait:
        return _login_page(f"Too many attempts. Try again in {wait // 60 + 1} min.")
    if not auth.check(ip, request.form.get("password")):
        return _login_page("Wrong password.")
    session.clear()
    session.permanent = True
    session["auth"] = auth.session_token()
    nxt = request.form.get("next", "/")
    return redirect(nxt if nxt.startswith("/") and not nxt.startswith("//") else "/")


@app.get("/logout")
def logout():
    session.clear()
    return redirect("/login" if auth.enabled() else "/")


@app.get("/api/me")
def api_me():
    return jsonify({"auth": auth.enabled()})

_lock = threading.Lock()
_frames = {}          # symbol -> candle DataFrame on a full 15m grid (Bybit USDT perpetuals)
_scan_cache = {}      # (threshold, window_hours) -> response dict
_loaded_at = None
_download = {"running": False, "done": 0, "total": 0, "current": None,
             "errors": [], "started": None, "finished": None, "message": None}
_live_cache = None    # (loaded_at, model mtime) -> scored rows
_live_lock = threading.Lock()  # one scoring pass at a time
_train = {"running": False, "started": None, "finished": None, "message": None}
_io_lock = threading.Lock()    # CSV writers: full download vs. live appends
_alerts = {}                   # id -> alert dict
_live_state = {"enabled": True, "running": False, "last_scan": None, "next_scan": None,
               "duration": None, "new": 0, "message": "Starting…"}
ALERT_KEEP_H = 72
_trend_state = {"running": False, "day": None, "message": None, "finished": None}
_ticks = {}                    # symbol -> {"bars": [live 15m bars after the last stored candle], "at": time}
LIVE_PRICE_S = 30
_pp = {"flags": {}, "ready": False, "record": None, "record_key": None, "base": None}  # pre-pump watch
_bo = {"data": None, "running": False, "seen": set()}
_rd = {"data": None, "running": False, "seen": None, "bar": None, "message": "Starting…"}            # big-move radar                                   # blow-off watch


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_frames():
    global _frames, _loaded_at, _live_cache
    paths = scanner.candle_paths()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        loaded = dict(zip((p.stem for p in paths), pool.map(signals.load_full, paths)))
    with _lock:
        _frames = {s: df for s, df in loaded.items() if df is not None}
        _scan_cache.clear()
        _loaded_at = time.time()
        _live_cache = None
    print(f"Loaded {len(_frames)} pairs in {time.time() - t0:.1f}s")
    backfill_alerts()
    threading.Thread(target=build_prepump, daemon=True).start()
    threading.Thread(target=refresh_blowoff, daemon=True).start()
    if _rd["data"] is None:
        threading.Thread(target=refresh_radar, daemon=True).start()
    if carry.data()["data"] is None:
        threading.Thread(target=carry.refresh, daemon=True).start()
    # Score the latest candles in the background so the Signals page opens fast
    if signals.MODEL_PATH.exists():
        threading.Thread(target=get_live, daemon=True).start()


def frame_for(symbol, _exchange=None):
    with _lock:
        return _frames.get(symbol)


def ts(t):
    return int(t.timestamp())


def clean(x):
    return None if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) else x


def run_scan(threshold, window_hours):
    key = (threshold, window_hours)
    with _lock:
        if key in _scan_cache:
            return _scan_cache[key]
        frames = dict(_frames)

    events, n_candles, n_labeled = [], 0, 0
    dmin = dmax = None
    for symbol, raw in frames.items():
        df = scanner.label_frame(raw, threshold, window_hours)
        n_candles += int(df["close"].notna().sum())
        n_labeled += int(df["label"].sum())
        dmin = df.index[0] if dmin is None else min(dmin, df.index[0])
        dmax = df.index[-1] if dmax is None else max(dmax, df.index[-1])
        events.extend(scanner.find_events(symbol, df, threshold, window_hours))

    rows = [{
        "symbol": e["symbol"],
        "first_signal": ts(e["first_signal"]),
        "entry_time": ts(e["entry_time"]),
        "entry_price": e["entry_price"],
        "peak_time": ts(e["peak_time"]),
        "peak_price": e["peak_price"],
        "gain": e["max_gain_pct"],
        "close_gain": clean(e["max_close_gain_pct"]),
        "hours_to_threshold": e["hours_to_threshold"],
        "hours_to_peak": e["hours_to_peak"],
        "labeled_candles": e["labeled_candles"],
        "volume_before": e["quote_vol_24h_before"],
    } for e in events]
    rows.sort(key=lambda r: -r["gain"])

    result = {
        "threshold": threshold,
        "window_hours": window_hours,
        "pairs": len(frames),
        "candles": n_candles,
        "labeled_candles": n_labeled,
        "start": ts(dmin) if dmin is not None else None,
        "end": ts(dmax) if dmax is not None else None,
        "loaded_at": int(_loaded_at or 0),
        "events": rows,
    }
    with _lock:
        _scan_cache[key] = result
    return result


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/")
def index():
    resp = send_from_directory(app.static_folder, "index.html")
    resp.headers["Cache-Control"] = "no-cache"  # always check for a newer page after an update
    return resp


@app.get("/api/scan")
def api_scan():
    try:
        threshold = round(float(request.args.get("threshold", 0.30)), 4)
        window = int(request.args.get("window", 24))
    except ValueError:
        abort(400)
    if not (0.01 <= threshold <= 20 and 1 <= window <= 168):
        abort(400)
    return jsonify(run_scan(threshold, window))


@app.get("/api/symbols")
def api_symbols():
    with _lock:
        return jsonify(sorted(_frames))


@app.get("/api/candles/<symbol>")
def api_candles(symbol):
    df = frame_for(symbol)
    if df is None:
        abort(404)
    start = request.args.get("from", type=int)
    end = request.args.get("to", type=int)
    if start is not None:
        df = df[df.index >= pd.Timestamp(start, unit="s", tz="UTC")]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end, unit="s", tz="UTC")]
    df = df.dropna(subset=["close"])
    t = df.index.as_unit("s").astype("int64").tolist()
    return jsonify({
        "symbol": symbol,
        "t": t,
        "o": df["open"].tolist(), "h": df["high"].tolist(),
        "l": df["low"].tolist(), "c": df["close"].tolist(),
        "v": df["quote_volume"].tolist(),
    })


@app.get("/api/download")
def api_download_status():
    with _lock:
        return jsonify(_download)


@app.post("/api/download")
def api_download_start():
    days = int((request.get_json(silent=True) or {}).get("days", 90))
    with _lock:
        if _download["running"]:
            return jsonify(_download), 409
        _download.update(running=True, done=0, total=0, current=None, errors=[],
                         started=time.time(), finished=None, message="Connecting to Bybit…")
    threading.Thread(target=_download_worker, args=(days,), daemon=True).start()
    return jsonify(_download), 202


def _download_worker(days):
    def progress(done, total, symbol, error):
        with _lock:
            _download.update(done=done, total=total, current=symbol,
                             message=f"Downloading {symbol}")
            if error:
                _download["errors"].append(f"{symbol}: {error}")

    try:
        with _io_lock:
            scanner.download_all(days, workers=8, progress=progress)
        with _lock:
            _download["message"] = "Reloading data…"
        load_frames()
        message = "Up to date"
    except BaseException as e:  # scanner uses sys.exit on connection failure
        message = f"Download failed: {e}"
    with _lock:
        _download.update(running=False, finished=time.time(), message=message)


# --------------------------------------------------------------------------- #
# Signals
# --------------------------------------------------------------------------- #
@app.get("/api/signals/report")
def api_signal_report():
    if not signals.REPORT_PATH.exists():
        return jsonify(None)
    return app.response_class(signals.REPORT_PATH.read_text(), mimetype="application/json")


def get_live():
    """Latest-candle scores for every pair, recomputed when data or model change (~10s)."""
    global _live_cache
    with _live_lock:
        key = (_loaded_at, signals.MODEL_PATH.stat().st_mtime)
        with _lock:
            cached = _live_cache
            frames = dict(_frames)
        if not cached or cached[0] != key:
            t0 = time.time()
            rows = signals.live_scores(frames)
            out = [{k: (ts(v) if k == "time" else clean(v)) for k, v in r.items()} for r in rows]
            cached = (key, out)
            with _lock:
                _live_cache = cached
            print(f"Scored {len(out)} pairs in {time.time() - t0:.1f}s")
        return cached[1]


@app.get("/api/signals/live")
def api_signal_live():
    if not signals.MODEL_PATH.exists():
        return jsonify(None)
    return jsonify({"scored_at": int(time.time()), "rows": get_live()})


@app.get("/api/signals/train")
def api_train_status():
    with _lock:
        return jsonify(_train)


@app.post("/api/signals/train")
def api_train_start():
    with _lock:
        if _train["running"]:
            return jsonify(_train), 409
        _train.update(running=True, started=time.time(), finished=None,
                      message="Building features and training (about 2 minutes)…")
    threading.Thread(target=_train_worker, daemon=True).start()
    return jsonify(_train), 202


def _train_worker():
    try:
        signals.run(threshold=0.30, test_days=28)
        get_live()
        message = "Model retrained"
    except Exception as e:
        message = f"Training failed: {e}"
    with _lock:
        _train.update(running=False, finished=time.time(), message=message)


# --------------------------------------------------------------------------- #
# Live alerts
# --------------------------------------------------------------------------- #
def _add_alerts(items, fresh_after=None):
    """Store alerts; those whose candle closed after `fresh_after` count as live (new)."""
    new = []
    with _lock:
        for a in items:
            if a["id"] in _alerts:
                continue
            fresh = fresh_after is not None and a["time"] >= fresh_after
            a["source"] = "live" if fresh else "history"
            a["detected_at"] = time.time() if fresh else None
            _alerts[a["id"]] = a
            if fresh:
                new.append(a)
        cutoff = time.time() - ALERT_KEEP_H * 3600
        for k in [k for k, a in _alerts.items() if a["time"] < cutoff]:
            del _alerts[k]
    return new


def backfill_alerts(hours=48):
    """Recreate the last `hours` of alerts from stored candles (so the feed isn't empty)."""
    with _lock:
        frames = dict(_frames)
    since = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)
    items = [a for sym, df in frames.items() for a in live.alerts_for(sym, df, since)]
    _add_alerts(items)


def run_live_cycle():
    global _loaded_at, _live_cache
    t0 = time.time()
    with _io_lock:
        with _lock:
            frames = dict(_frames)

        def one(item):
            sym, df = item
            last = df["close"].last_valid_index()
            try:
                rows = live.fetch_since(sym, last)
                return sym, (live.append_rows(sym, df, rows) if rows else None), last
            except Exception as e:  # delisted pair, transient error — skip this round
                print(f"live: {sym} failed ({e})")
                return sym, None, last

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(one, frames.items()))

    updated = {s: df for s, df, _ in results if df is not None}
    if updated:
        with _lock:
            _frames.update(updated)
            _scan_cache.clear()
            _loaded_at = time.time()
            _live_cache = None
    # Alerts on candles that closed within the last 45 min are "live"; older ones
    # (catching up after downtime) are filed as history so they don't notify.
    fresh_after = int(time.time()) - 45 * 60 - 900
    items = [a for s, df, last in results if df is not None for a in live.alerts_for(s, df, last)]
    fresh = _add_alerts(items, fresh_after)
    new = len(fresh)
    _telegram_after_scan(fresh)
    try:
        update_prepump(results, fresh_after)
        if updated:
            threading.Thread(target=refresh_blowoff, args=(fresh_after,), daemon=True).start()
    except Exception as e:
        print(f"pre-pump watch: {e}")
    if updated and signals.MODEL_PATH.exists():
        threading.Thread(target=get_live, daemon=True).start()
    return len(updated), new, time.time() - t0


def _telegram_after_scan(fresh):
    """Send new alerts and newly decided 1h checks to Telegram, per the user's settings."""
    if not notify.config():
        return
    if fresh:
        batch = [a for a in fresh if notify.wants(a, "new")]
        if batch:
            notify.send(notify.format_alerts(batch, "new"))
    # 1h follow-through: check live alerts whose hour has just finished
    with _lock:
        pending = [a for a in _alerts.values() if a.get("source") == "live" and not a.get("confirm_sent")
                   and a["time"] >= time.time() - 6 * 3600]
    confirmed = []
    for a in pending:
        df = frame_for(a["symbol"], a.get("exchange"))
        t = pd.Timestamp(a["time"], unit="s", tz="UTC")
        if df is None or t not in df.index:
            continue
        c = setups.confirmation(df, t)
        if c is None:
            continue  # the hour isn't over yet
        a["confirm_sent"] = True
        if c["confirmed"]:
            confirmed.append({**a, "chg_1h": c["chg_1h"],
                              "tags": exchange.symbols().get(a["symbol"], {}).get("tags", [])})
    batch = [a for a in confirmed if notify.wants(a, "confirmed")]
    if batch:
        notify.send(notify.format_alerts(batch, "confirmed"))


@app.get("/api/telegram")
def api_telegram():
    return jsonify(notify.public_config())


@app.post("/api/telegram")
def api_telegram_update():
    body = request.get_json(silent=True) or {}
    if body.get("test"):
        ok = notify.send("🧪 Test message from Pump Scanner — Telegram alerts are working.")
        return jsonify({"sent": ok, **notify.public_config()})
    try:
        return jsonify(notify.update({k: body[k] for k in notify.DEFAULTS if k in body}))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


def _next_boundary():
    # 12s after the next 15m close, so the exchange has the finished candle
    return (time.time() // live.INTERVAL_S + 1) * live.INTERVAL_S + 12


def _live_loop():
    first = True
    while True:
        due = first or (_live_state["next_scan"] and time.time() >= _live_state["next_scan"])
        if _live_state["enabled"] and due and not _download["running"]:
            first = False
            _live_state.update(running=True, message="Scanning…")
            try:
                n_upd, new, dur = run_live_cycle()
                _live_state.update(last_scan=time.time(), duration=round(dur, 1), new=new,
                                   message=f"Updated {n_upd} pairs, {new} new alert{'s' if new != 1 else ''}")
            except BaseException as e:
                _live_state["message"] = f"Scan failed: {e}"
            _live_state.update(running=False, next_scan=_next_boundary())
        elif _live_state["enabled"] and not _live_state["next_scan"]:
            _live_state["next_scan"] = _next_boundary()
        cd = carry.data()["data"]
        if cd and time.time() - cd["built"] > 3600 and not carry.data()["running"]:
            threading.Thread(target=carry.refresh, daemon=True).start()
        bar = int(time.time() // 14400)
        if _rd["bar"] is not None and bar != _rd["bar"] and time.time() - bar * 14400 > 90 and not _rd["running"]:
            threading.Thread(target=refresh_radar, daemon=True).start()
        if _trend_due():
            threading.Thread(target=refresh_trend, daemon=True).start()
        time.sleep(2)


def _setup_stats(path=setups.REPORT_PATH):
    if not path.exists():
        return None
    rep = json.loads(path.read_text())
    return {name: {**s["all"], "per_day": s.get("per_day"),
                   "best_sim": max(s["sims"], key=lambda x: x["mean"])}
            for name, s in rep["setups"].items()}


@app.get("/api/alerts")
def api_alerts():
    with _lock:
        alerts = [dict(a) for a in _alerts.values()]
        frames = dict(_frames)
    meta = exchange.symbols()
    for a in alerts:
        df = frames.get(a["symbol"])
        a.update({k: clean(v) for k, v in live_adjust(live.outcome(df, a), a).items()} if df is not None else {})
        a["tags"] = meta.get(a["symbol"], {}).get("tags", [])
    alerts.sort(key=lambda a: (-a["time"], a["symbol"]))
    return jsonify({"live": _live_state, "now": time.time(), "alerts": alerts, "stats": _setup_stats(),
                    "coins": len(frames)})


# --------------------------------------------------------------------------- #
# Live prices between 15m scans (1m candles for coins with an open alert or flag)
# --------------------------------------------------------------------------- #
def fetch_live(symbol):
    """Live 15m bars after the last stored candle, cached for 15s."""
    tk = _ticks.get(symbol)
    if tk and time.time() - tk["at"] < 15:
        return tk
    df = frame_for(symbol)
    if df is None:
        return None
    last = df["close"].last_valid_index()
    start_ms = int(last.timestamp() * 1000) + 900_000
    if start_ms < (time.time() - 6 * 3600) * 1000:
        return None  # stored candles are far behind; the next scan catches up
    tk = {"bars": exchange.live_bars(symbol, start_ms), "at": time.time()}
    _ticks[symbol] = tk
    return tk


def _live_prices_loop():
    while True:
        cutoff = time.time() - 25 * 3600
        with _lock:
            syms = {a["symbol"] for a in _alerts.values() if a["time"] >= cutoff}
            syms |= {f["symbol"] for f in _pp["flags"].values() if f["time"] >= cutoff}
            if _rd["data"]:
                syms |= {p["symbol"] for p in _rd["data"]["picks"] if p["time"] >= cutoff}
            if _bo["data"]:
                syms |= {x["symbol"] for x in _bo["data"]["items"] if x["status"] in ("watching", "open")}
        for sym in [k for k in _ticks if k not in syms]:
            _ticks.pop(sym, None)
        if not _live_state["running"]:
            with ThreadPoolExecutor(max_workers=6) as pool:
                list(pool.map(lambda s: _safe(fetch_live, s), syms))
        time.sleep(LIVE_PRICE_S)


def _safe(fn, *a):
    try:
        return fn(*a)
    except Exception:
        return None


def live_adjust(res, item):
    """Move an alert's/flag's result on to the live price (between 15m candle closes)."""
    tk = _ticks.get(item["symbol"])
    if not tk or not tk["bars"] or time.time() - tk["at"] > 3 * LIVE_PRICE_S:
        return res
    entry, end = item["price"], item["time"] + 900 + 24 * 3600
    res["now"] = tk["bars"][-1]["close"] / entry - 1
    res["hours"] = (time.time() - item["time"] - 900) / 3600
    res["live_at"] = tk["at"]
    inside = [b for b in tk["bars"] if b["time"] < end]
    if inside and not res.get("done"):
        hi = max(b["high"] for b in inside) / entry - 1
        lo = min(b["low"] for b in inside) / entry - 1
        res["peak"] = hi if res.get("peak") is None else max(res["peak"], hi)
        res["trough"] = lo if res.get("trough") is None else min(res["trough"], lo)
        if "reached5" in res:
            res.update(reached5=res["peak"] >= 0.05, reached10=res["peak"] >= 0.10)
    return res


@app.get("/api/live_bars/<symbol>")
def api_live_bars(symbol):
    try:
        tk = fetch_live(symbol.upper())
    except Exception as e:
        return jsonify({"bars": [], "error": str(e)})
    return jsonify({"bars": (tk or {}).get("bars", []), "at": (tk or {}).get("at")})


# --------------------------------------------------------------------------- #
# Pre-pump watch
# --------------------------------------------------------------------------- #
def build_prepump():
    """All flags in the stored candles (for the track record), plus base rates."""
    with _lock:
        frames = dict(_frames)
    items = prepump.fill_ls([f for s, df in frames.items() for f in prepump.flags(s, df)])
    base = prepump.base_rates(frames)
    with _lock:
        _pp["flags"].update({f["id"]: f for f in items})
        _pp.update(ready=True, base=base, record_key=None)
    print(f"Pre-pump watch: {len(items)} flags in stored candles")


def update_prepump(results, fresh_after):
    if not _pp["ready"]:
        return
    with _lock:
        last = {}
        for f in _pp["flags"].values():
            last[f["symbol"]] = max(last.get(f["symbol"], 0), f["time"])
    new = []
    for sym, df, since in results:
        if df is None:
            continue
        for f in prepump.flags(sym, df.iloc[-prepump.TAIL:], since, last.get(sym)):
            fresh = f["time"] >= fresh_after
            f["ls"] = prepump.ls_at(sym, f["time"], live=fresh)
            f["source"] = "live" if fresh else "history"
            new.append(f)
    if not new:
        return
    with _lock:
        _pp["flags"].update({f["id"]: f for f in new})
    if _pp["base"] is None or time.time() - _pp["base"]["at"] > 86400:
        with _lock:
            frames = dict(_frames)
        _pp["base"] = prepump.base_rates(frames)
    meta = exchange.symbols()
    batch = [{**f, "tags": meta.get(f["symbol"], {}).get("tags", [])} for f in new if f["source"] == "live"]
    batch = [f for f in batch if notify.wants_prepump(f)]
    if batch:
        notify.send(notify.format_prepump(batch))


# --------------------------------------------------------------------------- #
# Blow-off watch (short side, paper-tracked)
# --------------------------------------------------------------------------- #
def refresh_blowoff(fresh_after=None):
    if _bo["running"]:
        return
    _bo["running"] = True
    try:
        with _lock:
            frames = dict(_frames)
        data = blowoff.scan_all(frames)
        first = _bo["data"] is None
        new = [x for x in data["items"] if x["entry_at"] and x["id"] not in _bo["seen"]
               and fresh_after is not None and x["entry_at"] >= fresh_after]
        _bo["seen"] |= {x["id"] for x in data["items"] if x["entry_at"]}
        _bo["data"] = data
        if not first and new:
            meta = exchange.symbols()
            batch = [{**x, "tags": meta.get(x["symbol"], {}).get("tags", [])} for x in new]
            batch = [x for x in batch if notify.wants_blowoff(x)]
            if batch:
                notify.send(notify.format_blowoff(batch, data["phase"]))
    except Exception as e:
        print(f"blow-off watch: {e}")
    finally:
        _bo["running"] = False


@app.get("/api/blowoff")
def api_blowoff():
    data = _bo["data"]
    if not data:
        return jsonify({"ready": False})
    days = min(float(request.args.get("days", 7)), 120)
    cutoff = time.time() - days * 86400
    meta = exchange.symbols()
    items = []
    for x in data["items"]:
        if x["event_at"] < cutoff and x["status"] not in ("watching", "open"):
            continue
        x = dict(x)
        tk = _ticks.get(x["symbol"])
        if tk and tk["bars"] and time.time() - tk["at"] < 3 * LIVE_PRICE_S:
            price = tk["bars"][-1]["close"]
            x["price_now"] = price
            if x["status"] == "open":
                x["result"] = (x["entry"] - price) / x["entry"] - blowoff.FEE
            elif x["status"] == "watching":
                x["off_top"] = price / x["top"] - 1
        x["tags"] = meta.get(x["symbol"], {}).get("tags", [])
        items.append({k: clean(v) if isinstance(v, float) else v for k, v in x.items()})
    items.sort(key=lambda x: -(x["entry_at"] or x["event_at"]))
    return jsonify({"ready": True, "now": time.time(), "items": items, "phase": data["phase"],
                    "record": data["record"], "rule": data["rule"], "built": data["built"]})


# --------------------------------------------------------------------------- #
# Big-move radar (4h model)
# --------------------------------------------------------------------------- #
def refresh_radar():
    if _rd["running"]:
        return
    _rd.update(running=True, bar=int(time.time() // 14400), message="Updating 4h candles…")
    try:
        syms = set(scanner.tracked_symbols())
        radar.update_store(sorted(syms | {"BTCUSDT"}))
        _rd["message"] = "Scoring coins…"
        data = radar.scan(syms)
        if data is None:
            _rd["message"] = "No model yet — run: python radar.py --train"
            return
        ids = {p["id"] for p in data["picks"]}
        if _rd["seen"] is not None:
            new = [p for p in data["picks"] if p["id"] not in _rd["seen"] and p["time"] >= time.time() - 5 * 3600]
            meta = exchange.symbols()
            batch = [{**p, "tags": meta.get(p["symbol"], {}).get("tags", [])} for p in new]
            if batch and notify.wants_radar():
                notify.send(notify.format_radar(batch, data["record"]))
        _rd.update(data=data, seen=ids, message=None)
    except Exception as e:
        _rd["message"] = f"Radar failed: {e}"
        print(f"radar: {e}")
    finally:
        _rd["running"] = False


@app.get("/api/radar")
def api_radar():
    data = _rd["data"]
    if not data:
        return jsonify({"ready": False, "message": _rd["message"], "running": _rd["running"]})
    days = min(float(request.args.get("days", 7)), 365)
    cutoff = time.time() - days * 86400
    meta = exchange.symbols()
    picks = []
    for p in data["picks"]:
        if p["time"] < cutoff:
            continue
        x = dict(p)
        x.update({k: clean(v) for k, v in live_adjust(dict(p), p).items()})
        x["tags"] = meta.get(p["symbol"], {}).get("tags", [])
        picks.append(x)
    picks.sort(key=lambda x: (-x["time"], -x["score"]))
    return jsonify({"ready": True, "now": time.time(), "picks": picks, "record": data["record"], "report": data["report"],
                    "built": data["built"], "running": _rd["running"], "message": _rd["message"]})


# --------------------------------------------------------------------------- #
# Why is it moving? (news, Bybit announcements, CoinGecko trending)
# --------------------------------------------------------------------------- #
@app.get("/api/why/<symbol>")
def api_why(symbol):
    symbol = symbol.upper()
    if symbol not in exchange.symbols():
        abort(404)
    return jsonify(news.why(symbol))


@app.get("/api/carry")
def api_carry():
    s = carry.data()
    return jsonify({"data": s["data"], "running": s["running"], "error": s["error"]})


@app.get("/api/listings")
def api_listings():
    """{symbol: Bybit launch time} — used to tag coins listed in the last 30 days."""
    return jsonify({s: m.get("listed") for s, m in exchange.symbols().items() if m.get("listed")})


@app.get("/api/trending")
def api_trending():
    return jsonify({"coins": news.trending(), "at": time.time()})


@app.get("/api/prepump")
def api_prepump():
    hours = min(float(request.args.get("hours", 72)), 24 * 30)
    with _lock:
        items = [dict(f) for f in _pp["flags"].values()]
        frames = dict(_frames)
        key = (_loaded_at, len(items))
    if not _pp["ready"]:
        return jsonify({"ready": False})
    if _pp["record_key"] != key:
        _pp.update(record=prepump.track_record(items, frames), record_key=key)
    cutoff = time.time() - hours * 3600
    meta = exchange.symbols()
    recent = []
    for f in items:
        df = frames.get(f["symbol"])
        if f["time"] < cutoff or df is None:
            continue
        f.update({k: clean(v) for k, v in live_adjust(prepump.outcome(df, f), f).items()})
        f["tags"] = meta.get(f["symbol"], {}).get("tags", [])
        recent.append(f)
    recent.sort(key=lambda f: (-f["time"], f["symbol"]))
    return jsonify({"ready": True, "now": time.time(), "flags": recent, "record": _pp["record"],
                    "base": _pp["base"], "rule": prepump.RULE})


# --------------------------------------------------------------------------- #
# Daily trend
# --------------------------------------------------------------------------- #
def refresh_trend():
    """Fetch new daily candles and rebuild the trend report (~40s first time, ~10s after)."""
    if _trend_state["running"]:
        return
    _trend_state.update(running=True, message="Updating daily candles…")
    btc_state = lambda r: next((x["in_trend"] for x in (r or {}).get("current", {}).get("rows", []) if x["symbol"] == "BTCUSDT"), None)
    before = btc_state(json.loads(trend.REPORT_PATH.read_text())) if trend.REPORT_PATH.exists() else None
    try:
        exchange.refresh_symbols()
        trend.download_all()
        _trend_state["message"] = "Rebuilding trend report…"
        report = trend.run()
        after = btc_state(report)
        cfg = notify.config()
        if cfg and cfg["enabled"] and cfg["trend_changes"] and before is not None and after is not None and after != before:
            btc = next(x for x in report["current"]["rows"] if x["symbol"] == "BTCUSDT")
            notify.send(("📈 <b>BTC entered a daily uptrend</b>" if after else "📉 <b>BTC left its daily uptrend</b>") +
                        f" (EMA{report['params']['fast']}/{report['params']['slow']} rule, close {btc['close']:,.0f}). "
                        f"In the backtest, holding BTC only during uptrends cut the worst drop from "
                        f"{report['btc']['max_dd'] * 100:.0f}% to {report['btc_trend']['max_dd'] * 100:.0f}%. Not advice.")
        _trend_state.update(day=time.strftime("%Y-%m-%d", time.gmtime()), message="Up to date")
    except BaseException as e:
        _trend_state["message"] = f"Trend update failed: {e}"
    _trend_state.update(running=False, finished=time.time())


def _trend_due():
    """Once per UTC day, a minute after the daily candle closes (or if the report is stale)."""
    today = time.strftime("%Y-%m-%d", time.gmtime())
    if _trend_state["running"] or _trend_state["day"] == today:
        return False
    if trend.REPORT_PATH.exists():
        end = json.loads(trend.REPORT_PATH.read_text()).get("end")
        yesterday = time.strftime("%Y-%m-%d", time.gmtime(time.time() - 86400))
        if end and end >= yesterday:
            _trend_state["day"] = today  # already has yesterday's close
            return False
    return time.gmtime().tm_hour * 60 + time.gmtime().tm_min >= 1


@app.get("/api/trend")
def api_trend():
    rep = json.loads(trend.REPORT_PATH.read_text()) if trend.REPORT_PATH.exists() else None
    return jsonify({"report": rep, "state": _trend_state})


@app.post("/api/trend/refresh")
def api_trend_refresh():
    threading.Thread(target=refresh_trend, daemon=True).start()
    return jsonify(_trend_state), 202


_quote_cache = {}  # symbol -> (fetched_at, quote)


@app.get("/api/quote/<symbol>")
def api_quote(symbol):
    """Live price and contract details for the trade planner (cached 20s)."""
    if not symbol.isalnum() or symbol not in exchange.symbols():
        abort(404)
    hit = _quote_cache.get(symbol)
    if hit and time.time() - hit[0] < 20:
        return jsonify(hit[1])
    try:
        t = exchange.get("/v5/market/tickers", {"category": "linear", "symbol": symbol})["list"][0]
        i = exchange.get("/v5/market/instruments-info", {"category": "linear", "symbol": symbol})["list"][0]
        tiers = exchange.get("/v5/market/risk-limit", {"category": "linear", "symbol": symbol})["list"]
    except Exception as e:
        return jsonify({"error": str(e)}), 502
    q = {
        "symbol": symbol, "price": float(t["lastPrice"]), "mark": float(t["markPrice"]),
        "change_24h": float(t.get("price24hPcnt") or 0), "funding": float(t.get("fundingRate") or 0),
        "next_funding": int(t.get("nextFundingTime") or 0) // 1000,
        "funding_interval_h": int(i.get("fundingInterval") or 480) / 60,
        "max_leverage": float(i["leverageFilter"]["maxLeverage"]),
        "qty_step": float(i["lotSizeFilter"]["qtyStep"]), "min_qty": float(i["lotSizeFilter"]["minOrderQty"]),
        "min_notional": float(i["lotSizeFilter"].get("minNotionalValue") or 0),
        "tick": float(i["priceFilter"]["tickSize"]),
        # maintenance margin rate by position size (for the liquidation price)
        "tiers": sorted(({"limit": float(x["riskLimitValue"]), "mmr": float(x["maintenanceMargin"]),
                          "max_leverage": float(x["maxLeverage"])} for x in tiers), key=lambda x: x["limit"]),
        "tags": exchange.symbols().get(symbol, {}).get("tags", []),
    }
    _quote_cache[symbol] = (time.time(), q)
    return jsonify(q)


@app.get("/api/collector")
def api_collector():
    return jsonify(collector.status)


@app.get("/api/journal")
def api_journal():
    return jsonify(journal.all_trades())


@app.post("/api/journal")
def api_journal_add():
    body = request.get_json(silent=True) or {}
    trades = body.get("trades") or ([body["trade"]] if body.get("trade") else [])
    if not trades or len(trades) > 20000:
        return jsonify({"error": "send 1–20000 trades"}), 400
    try:
        added, skipped = journal.add(trades)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"added": added, "skipped": skipped, "total": len(journal.all_trades())})


@app.patch("/api/journal/<trade_id>")
def api_journal_update(trade_id):
    t = journal.update(trade_id, request.get_json(silent=True) or {})
    return (jsonify(t), 200) if t else (jsonify({"error": "not found"}), 404)


@app.delete("/api/journal/<trade_id>")
def api_journal_delete(trade_id):
    if trade_id == "all":
        return jsonify({"deleted": journal.delete(None)})
    return jsonify({"deleted": journal.delete(trade_id)})


_fut_cache = {}  # symbol -> (fetched_at, context)


@app.get("/api/futures/<symbol>")
def api_futures(symbol):
    """Live futures context for one coin, cached for a minute."""
    if not symbol.isalnum():
        abort(404)
    hit = _fut_cache.get(symbol)
    if hit and time.time() - hit[0] < 60:
        return jsonify(hit[1])
    try:
        ctx = futures.live_context(symbol)
    except Exception as e:
        return jsonify({"error": str(e)}), 502
    _fut_cache[symbol] = (time.time(), ctx)
    return jsonify(ctx)


@app.get("/api/daily/<symbol>")
def api_daily(symbol):
    path = trend.DAILY_DIR / f"{symbol}.csv"
    if not symbol.isalnum() or not path.exists():
        abort(404)
    d = pd.read_csv(path)
    t = pd.to_datetime(d["date"], utc=True).dt.as_unit("s").astype("int64").tolist()
    return jsonify({"symbol": symbol, "t": t, "o": d["open"].tolist(), "h": d["high"].tolist(),
                    "l": d["low"].tolist(), "c": d["close"].tolist(), "v": d["quote_volume"].tolist()})


@app.post("/api/live")
def api_live_toggle():
    enabled = bool((request.get_json(silent=True) or {}).get("enabled", True))
    _live_state["enabled"] = enabled
    _live_state["message"] = "Live scanning on" if enabled else "Paused"
    if enabled:
        _live_state["next_scan"] = _next_boundary()
    return jsonify(_live_state)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Pump Scanner dashboard")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to serve on a public server (needs a password)")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--set-password", action="store_true", help="set or change the login password, then exit")
    ap.add_argument("--setup-telegram", action="store_true", help="connect a Telegram bot for alerts, then exit")
    ap.add_argument("--set-key", metavar="NAME", help="save an API key privately (helius or etherscan), then exit")
    ap.add_argument("--no-collect", action="store_true", help="don't run the background data collector")
    args = ap.parse_args()

    if args.set_password:
        auth.set_password_cli()
        sys.exit()
    if args.setup_telegram:
        notify.setup_cli()
        sys.exit()
    if args.set_key:
        collector.set_key_cli(args.set_key)
        sys.exit()
    if args.host not in ("127.0.0.1", "localhost") and not auth.enabled():
        sys.exit("Refusing to serve on a public address without a password. Run: python app.py --set-password")

    load_frames()
    threading.Thread(target=_live_loop, daemon=True).start()
    threading.Thread(target=_live_prices_loop, daemon=True).start()
    if not args.no_collect:
        collector.start()
    shown = "127.0.0.1" if args.host in ("0.0.0.0", "localhost") else args.host
    print(f"Open http://{shown}:{args.port}" + ("  (login required)" if auth.enabled() else ""))
    try:
        from waitress import serve  # production server, if installed
        serve(app, host=args.host, port=args.port, threads=8)
    except ImportError:
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
