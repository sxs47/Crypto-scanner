"""
Paper trading: two fake-money accounts of $500 that follow fixed rules on live Bybit prices, so the plans can be
judged forward (no backtest) without risking money.

A  "plan"  — confirmed ATR_SM 4h Buys that pass the checks (not already +25% in 24h, not futures-led, no big
             unlock, >= $20M traded per day). The "market weak" check is off for this test (user's choice). 10% margin, 3x. Stop -5%; at +3% close half and
             move the stop to break-even; rest at +8%; close after 24h.
B  "fast"  — 15m: EMA7 crosses above EMA21/50/200 and the candle closes above all four (coins >= $5M/day).
             10% margin, 5x. Stop -3%; exit when EMA7 drops below EMA21 (closed 15m candle) or at +8%; max 24h.

Profit lock (both, since 2026-10-18): once a trade is +2%, the stop moves to +1%, then trails 2% under the best
price, so a +2% move is never turned into a loss and big moves can still run to the target.
Fees 0.055% of the position per side (Bybit taker). Funding is not included. Max 5 open trades per account,
one per coin. Prices are checked about every 30 s (last traded price), so fills are approximate.
State lives in data/paper.json.
"""

import json
import threading
import time

import numpy as np
import pandas as pd

import scanner

PATH = scanner.DATA_DIR / "paper.json"
START = 500.0
FEE = 0.00055
MAX_OPEN = 5
LOCK_AT, LOCK_MIN, TRAIL = 0.02, 0.01, 0.02   # at +2% the stop goes to +1%, then follows 2% under the best price
RULES = {
    "plan": {"name": "Plan · 4h ATR (3x)", "lev": 3, "margin": 0.10, "stop": 0.05, "tp1": 0.03, "tp2": 0.08, "max_h": 24},
    "fast": {"name": "Fast · 15m EMA (5x)", "lev": 5, "margin": 0.10, "stop": 0.03, "tp1": None, "tp2": 0.08, "max_h": 24},
}
_lock = threading.Lock()


def _new(key):
    return {"key": key, **RULES[key], "start": START, "cash": START, "open": [], "closed": [], "started_at": time.time()}


def load():
    try:
        s = json.loads(PATH.read_text())
    except (OSError, ValueError):
        s = {}
    for k in RULES:
        if k not in s:
            s[k] = _new(k)
        s[k].update({x: RULES[k][x] for x in RULES[k]})      # rules may change; history stays
    return s


def save(s):
    PATH.write_text(json.dumps(s, indent=1))


def reset():
    with _lock:
        s = {k: _new(k) for k in RULES}
        save(s)


def _equity(acc, px):
    eq = acc["cash"]
    for p in acc["open"]:
        price = px.get(p["symbol"], p["entry"])
        eq += p["margin"] + p["qty"] * (price - p["entry"])
    return eq


def _open(acc, sym, price, reason, now, ctx=None):
    if len(acc["open"]) >= MAX_OPEN or any(p["symbol"] == sym for p in acc["open"]):
        return
    margin = acc["margin"] * _equity(acc, {})
    margin = min(margin, acc["cash"])
    if margin < 1:
        return
    notional = margin * acc["lev"]
    qty = notional / price
    acc["cash"] -= margin + notional * FEE
    acc["open"].append({"symbol": sym, "entry": price, "qty": qty, "qty0": qty, "margin": margin, "opened": now,
                        "stop": price * (1 - acc["stop"]), "half_done": False, "reason": reason, "fees": notional * FEE,
                        "realized": 0.0, "best": price, "ctx": ctx or {}})


def _close(acc, p, price, part, why, now):
    q = p["qty"] * part
    m = p["margin"] * part
    pnl = q * (price - p["entry"]) - q * price * FEE
    acc["cash"] += m + pnl
    p["qty"] -= q
    p["margin"] -= m
    p["realized"] += pnl
    p["fees"] += q * price * FEE
    if part >= 0.999:
        total = p["realized"] - (p["qty0"] * p["entry"] * FEE)      # include the entry fee
        acc["closed"].append({"symbol": p["symbol"], "entry": p["entry"], "exit": price, "opened": p["opened"], "closed": now,
                              "pnl": round(total, 4), "pnl_pct_margin": total / (p["qty0"] * p["entry"] / acc["lev"]),
                              "why": why, "reason": p["reason"], "ctx": p.get("ctx", {}),
                              "best_pct": p.get("best", price) / p["entry"] - 1})
        acc["closed"] = acc["closed"][-500:]
        return True
    return False


def manage(s, px, fast_exit=None, now=None):
    """Stops, targets and time exits at the live prices."""
    now = now or time.time()
    fast_exit = fast_exit or set()
    for key, acc in s.items():
        keep = []
        for p in acc["open"]:
            price = px.get(p["symbol"])
            if price is None:
                keep.append(p); continue
            p["best"] = max(p.get("best", price), price)
            if p["best"] >= p["entry"] * (1 + LOCK_AT):           # profit lock: never give back a +2% move
                p["stop"] = max(p["stop"], p["entry"] * (1 + LOCK_MIN), p["best"] * (1 - TRAIL))
            done = False
            if price <= p["stop"]:
                done = _close(acc, p, price, 1.0, ("profit lock" if p["stop"] > p["entry"] else "stop") if not p["half_done"] else ("profit lock" if p["stop"] > p["entry"] else "break-even stop"), now)
            elif acc["tp1"] and not p["half_done"] and price >= p["entry"] * (1 + acc["tp1"]):
                _close(acc, p, price, 0.5, "half at +3%", now)
                p["half_done"], p["stop"] = True, p["entry"]
            if not done and price >= p["entry"] * (1 + acc["tp2"]):
                done = _close(acc, p, price, 1.0, f"target +{acc['tp2']:.0%}", now)
            if not done and key == "fast" and p["symbol"] in fast_exit:
                done = _close(acc, p, price, 1.0, "EMA7 < EMA21", now)
            if not done and now - p["opened"] >= acc["max_h"] * 3600:
                done = _close(acc, p, price, 1.0, "24h time exit", now)
            if not done:
                keep.append(p)
        acc["open"] = keep


def fast_signals(symbols, turnover):
    """(entries, exits) from the app's 15m candles: EMA7 crossing above EMA21/50/200 on the last closed candle."""
    entries, exits = [], set()
    for sym in symbols:
        path = scanner.CANDLE_DIR / f"{sym}.csv"
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, usecols=["open_time", "close"]).tail(400)
        except Exception:
            continue
        c = df["close"].astype(float).ffill()
        if len(c) < 260:
            continue
        e = {n: c.ewm(span=n, adjust=False).mean().to_numpy() for n in (7, 21, 50, 200)}
        cc = c.to_numpy()
        top = (e[7] > e[21]) & (e[7] > e[50]) & (e[7] > e[200])
        if e[7][-1] < e[21][-1]:
            exits.add(sym)
        if top[-1] and not top[-2] and cc[-1] > max(e[n][-1] for n in e) and turnover.get(sym, 0) >= 5e6:
            entries.append((sym, float(cc[-1])))
    return entries, exits


def tick(px, turnover, plan_candidates, fast_symbols, fast_due, context=None):
    """Called about every 30 s by the app. plan_candidates: [(signal id, symbol, price, reason)] already filtered."""
    with _lock:
        s = load()
        now = time.time()
        fast_entries, fast_exit = ([], set())
        if fast_due:
            fast_entries, fast_exit = fast_signals(fast_symbols, turnover)
        manage(s, px, fast_exit, now)
        seen = s["plan"].setdefault("seen", [])
        for sid, sym, price, reason in plan_candidates:      # each confirmed Buy is taken at most once
            if sid in seen:
                continue
            seen.append(sid)
            _open(s["plan"], sym, px.get(sym, price), reason, now, context(sym, px.get(sym, price)) if context else None)
        s["plan"]["seen"] = seen[-2000:]
        for sym, price in fast_entries:
            _open(s["fast"], sym, px.get(sym, price), "15m EMA cross", now, context(sym, px.get(sym, price)) if context else None)
        save(s)
        return s


def summary(px):
    with _lock:
        s = load()
    out = {}
    for key, acc in s.items():
        eq = _equity(acc, px)
        closed = acc["closed"]
        wins = [t for t in closed if t["pnl"] > 0]
        peak, dd, run = acc["start"], 0.0, acc["start"]
        for t in closed:
            run += t["pnl"]
            peak = max(peak, run)
            dd = min(dd, run / peak - 1)
        out[key] = {
            "name": acc["name"], "start": acc["start"], "equity": eq, "cash": acc["cash"], "started_at": acc["started_at"],
            "trades": len(closed), "win_rate": len(wins) / len(closed) if closed else None,
            "avg_win": float(np.mean([t["pnl"] for t in wins])) if wins else None,
            "avg_loss": float(np.mean([t["pnl"] for t in closed if t["pnl"] <= 0])) if len(wins) < len(closed) else None,
            "max_drawdown": dd,
            "open": [{**p, "now": px.get(p["symbol"]), "pnl_now": p["realized"] + p["qty"] * (px.get(p["symbol"], p["entry"]) - p["entry"])}
                     for p in acc["open"]],
            "closed": closed[-40:][::-1],
            "rules": {k: acc[k] for k in ("lev", "margin", "stop", "tp1", "tp2", "max_h")},
        }
    return out
