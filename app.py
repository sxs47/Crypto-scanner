"""
Local web dashboard for the pump scanner.

    python app.py            # then open http://127.0.0.1:5000

Candle CSVs are loaded into memory once; scans for a given threshold/window
are computed on demand and cached. The "Update data" button runs the same
incremental Binance download as scanner.py in a background thread. The
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
from flask import Flask, abort, jsonify, request, send_from_directory

import bybit
import futures
import live
import scanner
import setups
import signals
import trend

# Some Binance symbols are non-ASCII (e.g. 币安人生USDT); Windows consoles default to cp1252
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

app = Flask(__name__, static_folder="static", static_url_path="/static")

_lock = threading.Lock()
_frames = {}          # symbol -> candle DataFrame on a full 15m grid (Binance)
_bframes = {}         # same for Bybit-only coins (Alerts → Bybit tab)
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
    load_bybit_frames()
    backfill_alerts()
    # Score the latest candles in the background so the Signals page opens fast
    if signals.MODEL_PATH.exists():
        threading.Thread(target=get_live, daemon=True).start()


def load_bybit_frames():
    global _bframes
    paths = bybit.candle_paths()
    with ThreadPoolExecutor(max_workers=8) as pool:
        loaded = dict(zip((p.stem for p in paths), pool.map(signals.load_full, paths)))
    with _lock:
        _bframes = {s: df for s, df in loaded.items() if df is not None}
    print(f"Loaded {len(_bframes)} Bybit-only pairs")


def refresh_bybit():
    """Refresh the Bybit-only coin list (new listings, volume filter) and fill any gaps."""
    try:
        with _io_lock:
            bybit.download_all()
        load_bybit_frames()
        backfill_alerts(exchanges=("bybit",))
    except Exception as e:
        print(f"Bybit refresh failed: {e}")


def frame_for(symbol, exchange=None):
    with _lock:
        if exchange == "bybit":
            return _bframes.get(symbol)
        return _frames.get(symbol) if symbol in _frames else _bframes.get(symbol)


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
    return send_from_directory(app.static_folder, "index.html")


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
                         started=time.time(), finished=None, message="Connecting to Binance…")
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
    new = 0
    with _lock:
        for a in items:
            if a["id"] in _alerts:
                continue
            fresh = fresh_after is not None and a["time"] >= fresh_after
            a["source"] = "live" if fresh else "history"
            a["detected_at"] = time.time() if fresh else None
            _alerts[a["id"]] = a
            new += fresh
        cutoff = time.time() - ALERT_KEEP_H * 3600
        for k in [k for k, a in _alerts.items() if a["time"] < cutoff]:
            del _alerts[k]
    return new


def backfill_alerts(hours=48, exchanges=("binance", "bybit")):
    """Recreate the last `hours` of alerts from stored candles (so the feed isn't empty)."""
    with _lock:
        sources = {"binance": dict(_frames), "bybit": dict(_bframes)}
    since = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)
    items = [a for ex in exchanges for sym, df in sources[ex].items() for a in live.alerts_for(sym, df, since, ex)]
    _add_alerts(items)


def run_live_cycle():
    global _loaded_at, _live_cache
    t0 = time.time()
    with _io_lock:
        live.ensure_base_url()
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

        # Bybit-only coins: same rules, Bybit's kline endpoint
        with _lock:
            bframes = dict(_bframes)
        now_ms = int(time.time() * 1000)
        end_ms = now_ms - now_ms % (live.INTERVAL_S * 1000)

        def one_bybit(item):
            sym, df = item
            last = df["close"].last_valid_index()
            try:
                rows = bybit.fetch_klines(sym, int(last.timestamp() * 1000) + live.INTERVAL_S * 1000, end_ms)
                return sym, (live.append_rows(sym, df, rows, bybit.CANDLE_DIR) if rows else None), last
            except Exception as e:
                print(f"live: Bybit {sym} failed ({e})")
                return sym, None, last

        with ThreadPoolExecutor(max_workers=8) as pool:
            bresults = list(pool.map(one_bybit, bframes.items()))
    bupdated = {s: df for s, df, _ in bresults if df is not None}
    if bupdated:
        with _lock:
            _bframes.update(bupdated)

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
    items += [a for s, df, last in bresults if df is not None for a in live.alerts_for(s, df, last, "bybit")]
    new = _add_alerts(items, fresh_after)
    if updated and signals.MODEL_PATH.exists():
        threading.Thread(target=get_live, daemon=True).start()
    return len(updated) + len(bupdated), new, time.time() - t0


def _next_boundary():
    # 12s after the next 15m close, so Binance has the finished candle
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
        if _trend_due():
            threading.Thread(target=refresh_trend, daemon=True).start()
            threading.Thread(target=refresh_bybit, daemon=True).start()
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
        sources = {"binance": dict(_frames), "bybit": dict(_bframes)}
    meta = bybit.symbols()
    for a in alerts:
        ex = a.get("exchange", "binance")
        df = sources[ex].get(a["symbol"])
        a.update({k: clean(v) for k, v in live.outcome(df, a).items()} if df is not None else {})
        if ex == "bybit":
            a["tags"] = meta.get(a["symbol"], {}).get("tags", [])
    alerts.sort(key=lambda a: (-a["time"], a["symbol"]))
    return jsonify({"live": _live_state, "now": time.time(), "alerts": alerts,
                    "stats": _setup_stats(), "stats_bybit": _setup_stats(setups.BYBIT_REPORT_PATH),
                    "bybit_coins": len(_bframes)})


# --------------------------------------------------------------------------- #
# Daily trend
# --------------------------------------------------------------------------- #
def refresh_trend():
    """Fetch new daily candles and rebuild the trend report (~40s first time, ~10s after)."""
    if _trend_state["running"]:
        return
    _trend_state.update(running=True, message="Updating daily candles…")
    try:
        trend.download_all()
        _trend_state["message"] = "Rebuilding trend report…"
        trend.run()
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
    load_frames()
    threading.Thread(target=_live_loop, daemon=True).start()
    print("Open http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
