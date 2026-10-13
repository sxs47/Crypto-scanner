"""
Funding carry (cash-and-carry): buy a coin on Bybit spot and short the same amount on the perpetual.
Price moves cancel out; you collect the funding the perp shorts are paid (or pay it when funding is negative).

Research (Bybit funding history 2021–2026): BTC paid shorts +38%, +3%, +9%, +12%, +5% per year (2021→2025);
a weekly rotation into the 5 coins with the highest funding last week made +59%, −1%, +5%, +17%, +5% after
switching costs, and +12% in 2026 to mid-October. Risks: the short can be liquidated on a big spike if its margin
is too thin, funding turns negative at times, and everything sits on one exchange.
"""

import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import exchange

MIN_TURNOVER = 5_000_000       # only coins liquid enough for both legs
TOP = 5
TOP_TRADED = 60             # rotation picks only from the 60 most-traded perpetuals (as in the research)
SWITCH_COST = 0.0031           # spot buy+sell 0.1% each + perp open+close 0.055% each
RESEARCH = {"btc": {"2021": 0.383, "2022": 0.029, "2023": 0.089, "2024": 0.120, "2025": 0.051, "2026*": 0.019},
            "rotation": {"2021": 0.590, "2022": -0.010, "2023": 0.047, "2024": 0.165, "2025": 0.047, "2026*": 0.120}}
_state = {"data": None, "running": False, "error": None}
_hist = {}                     # symbol -> list of (ms, rate) for the last ~35 days
_lock = threading.Lock()


def spot_bases():
    rows, cursor = [], ""
    while True:
        res = exchange.get("/v5/market/instruments-info", {"category": "spot", "limit": 1000, "cursor": cursor})
        rows += res["list"]
        cursor = res.get("nextPageCursor") or ""
        if not cursor:
            break
    return {x["baseCoin"] for x in rows if x["quoteCoin"] == "USDT" and x["status"] == "Trading"}


def base_of(symbol):
    b = symbol.removesuffix("USDT")
    return re.sub(r"^\d+", "", b) or b


def _history(sym, days=31):
    """Funding payments of the last `days` days, newest first from Bybit (paged)."""
    start = int((time.time() - days * 86400) * 1000)
    out, end = [], int(time.time() * 1000)
    for _ in range(10):
        b = exchange.get("/v5/market/funding/history", {"category": "linear", "symbol": sym, "startTime": start,
                                                        "endTime": end, "limit": 200})["list"]
        if not b:
            break
        out += [(int(x["fundingRateTimestamp"]), float(x["fundingRate"])) for x in b]
        oldest = min(t for t, _ in out)
        if len(b) < 200 or oldest <= start:
            break
        end = oldest - 1
    return sorted(set(out))


def _annual(hist, days):
    cut = (time.time() - days * 86400) * 1000
    r = [x for t, x in hist if t >= cut]
    if not r:
        return None, None
    return float(np.sum(r) * 365 / days), float(np.mean(np.array(r) < 0))


def refresh():
    if _state["running"]:
        return
    _state["running"] = True
    try:
        meta = exchange.symbols()
        spot = spot_bases()
        tick = {t["symbol"]: t for t in exchange.get("/v5/market/tickers", {"category": "linear"})["list"]}
        syms = [s for s in meta if float(tick.get(s, {}).get("turnover24h") or 0) >= MIN_TURNOVER]
        with ThreadPoolExecutor(6) as pool:
            for s, h in zip(syms, pool.map(lambda s: _safe(_history, s), syms)):
                if h:
                    _hist[s] = h
        rows = []
        for s in syms:
            t = tick.get(s, {})
            h = _hist.get(s, [])
            interval = meta[s].get("funding_interval_h", 8) or 8
            now_rate = float(t.get("fundingRate") or 0)
            a7, neg7 = _annual(h, 7)
            a30, neg30 = _annual(h, 30)
            rows.append({
                "symbol": s, "base": base_of(s), "spot": base_of(s) in spot,
                "funding_now": now_rate, "interval_h": interval,
                "now_yearly": now_rate * (24 / interval) * 365,
                "avg7_yearly": a7, "avg30_yearly": a30, "neg_share_30d": neg30,
                "next_funding": int(t.get("nextFundingTime") or 0) // 1000,
                "turnover24": float(t.get("turnover24h") or 0), "oi_value": float(t.get("openInterestValue") or 0),
                "price": float(t.get("lastPrice") or 0), "tags": meta[s].get("tags", []),
            })
        top_traded = {r["symbol"] for r in sorted(rows, key=lambda r: -r["turnover24"])[:TOP_TRADED]}
        for r in rows:
            r["top_traded"] = r["symbol"] in top_traded
        # same universe as the research: the most-traded coins, and the coin must also trade on Bybit spot
        eligible = [r for r in rows if r["spot"] and r["top_traded"] and r["avg7_yearly"] is not None and r["avg7_yearly"] > 0]
        picks = [r["symbol"] for r in sorted(eligible, key=lambda r: -r["avg7_yearly"])[:TOP]]
        # how last month's weekly rotation would have done (from the fetched 30-day history)
        weeks = []
        now_ms = time.time() * 1000
        for w in range(3, -1, -1):                       # 4 finished-ish weeks: pick on week w+1, earn on week w
            lo, mid, hi = now_ms - (w + 2) * 7 * 86400e3, now_ms - (w + 1) * 7 * 86400e3, now_ms - w * 7 * 86400e3
            prev = {r["symbol"]: sum(x for t, x in _hist.get(r["symbol"], []) if lo <= t < mid) for r in rows if r["spot"] and r["top_traded"]}
            chosen = [s for s, v in sorted(prev.items(), key=lambda kv: -kv[1]) if v > 0][:TOP]
            earn = np.mean([sum(x for t, x in _hist.get(s, []) if mid <= t < hi) for s in chosen]) if chosen else 0.0
            weeks.append({"start": int(mid / 1000), "picks": chosen, "earned": float(earn)})   # before switching costs
        _state.update(data={"rows": rows, "picks": picks, "weeks": weeks, "research": RESEARCH, "built": time.time(),
                            "switch_cost": SWITCH_COST, "min_turnover": MIN_TURNOVER}, error=None)
    except Exception as e:
        _state["error"] = str(e)
        print(f"carry: {e}")
    finally:
        _state["running"] = False


def _safe(fn, *a):
    try:
        return fn(*a)
    except Exception:
        return None


def data():
    return _state
