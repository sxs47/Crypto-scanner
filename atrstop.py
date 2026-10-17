"""
ATR_SM_BuySell (ATR Smoothed by dysrupt, modified by nifty_alchemist) — exact Pine v4 logic on 4h candles.
A trailing line follows the price at Multiplier × ATR(Period); Buy when the close crosses above it, Sell below.

Settings 5 / 5 were chosen on Bybit 4h data from 12 Jul–11 Sep 2026 (the best of 63 settings tested on 15m, 1h and
4h). On the unseen 12 Sep–12 Oct: 47% of Buys reached +10% before −5% and 34% reached +20% before −10%
(random moments: 32% / 23%; break-even 33%); the typical run before the Sell signal was ~+16%. Paper-tracked
from 12 Sep 2026 on. Not advice.
"""

import time

import numpy as np
import pandas as pd

import radar

PERIOD, MULT = 5, 5.0
TEST_START = pd.Timestamp("2026-09-12", tz="UTC")
GRID = ((0.10, 0.05), (0.20, 0.10))
MAXBARS = 42                      # follow each Buy for up to 7 days
H4 = pd.Timedelta(hours=4)


def trail_pos(c, h, l, period=PERIOD, mult=MULT):
    n = len(c)
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    tr[0] = h[0] - l[0]
    atr = pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean().to_numpy()      # Pine atr() = RMA
    loss = mult * atr
    trail, pos = np.zeros(n), np.zeros(n, dtype=np.int8)
    pt, pp = 0.0, 0
    for i in range(period + 1, n):
        ci, c1, nl = c[i], c[i - 1], loss[i]
        if ci > pt and c1 > pt:
            t = max(pt, ci - nl)
        elif ci < pt and c1 < pt:
            t = min(pt, ci + nl)
        else:
            t = ci - nl if ci > pt else ci + nl
        p = 1 if (c1 < pt and ci > pt) else (-1 if (c1 > pt and ci < pt) else pp)
        trail[i], pos[i] = t, p
        pt, pp = t, p
    return trail, pos


def state_at(k, t):
    """(long?, hours since the last Buy or None) at 4h candle t — used to tag Run picks."""
    c, h, l = (k[x].ffill().to_numpy() for x in ("c", "h", "l"))
    _, pos = trail_pos(c, h, l)
    i = k.index.get_loc(t)
    if pos[i] != 1:
        return False, None
    j = i
    while j > 0 and pos[j - 1] == 1:
        j -= 1
    return True, (i - j) * 4


def _one(sym, since):
    k = radar.load(sym)
    if k is None or len(k) < 200:
        return []
    k = k.ffill()
    c, h, l, q = (k[x].to_numpy() for x in ("c", "h", "l", "q"))
    trail, pos = trail_pos(c, h, l)
    n, out = len(c), []
    for i in np.flatnonzero((pos[1:] == 1) & (pos[:-1] != 1)) + 1:
        close_t = k.index[i] + H4
        if close_t < since:
            continue
        e = c[i]
        end = min(n, i + 1 + MAXBARS)
        res = {}
        for up, dn in GRID:
            st = "open"
            for j in range(i + 1, end):
                if l[j] <= e * (1 - dn):
                    st = "stopped"; break
                if h[j] >= e * (1 + up):
                    st = "hit"; break
            if st == "open" and i + MAXBARS < n:
                st = "neither"
            res[f"t{int(up*100)}_{int(dn*100)}"] = st
        best, j, sold_at = 0.0, i + 1, None
        while j < end and pos[j] == 1:
            best = max(best, h[j] / e - 1)
            j += 1
        if j < n and pos[j] != 1:
            sold_at = {"time": int((k.index[j] + H4).timestamp()), "ret": float(c[j] / e - 1)}
        out.append({"id": f"{sym}|{int(close_t.timestamp())}", "symbol": sym, "time": int(close_t.timestamp()) - 900,
                    "price": float(e), "line": float(trail[i]), "line_now": float(trail[-1]), "stop_dist": float((e - trail[i]) / e),
                    "best_until_sell": float(best), "sold": sold_at, "long_now": bool(sold_at is None and pos[-1] == 1),
                    "now": float(c[-1] / e - 1), "done": bool(sold_at is not None or i + MAXBARS < n), **res})
    return out


def signals(symbols, days=7):
    """ATR tab: every Buy/Sell flip on closed 4h candles in the last `days` (no volume filter), plus each coin's
    last trailing line, so the server can show 'blinking' flips on the forming candle from live prices."""
    now = pd.Timestamp.now(tz="UTC")
    since = now - pd.Timedelta(days=days)
    items, states = [], {}
    above_now, above_7d = [], []
    for s in sorted(symbols):
        try:
            k = radar.load(s)
            if k is None or len(k) < 50 or k.index[-1] < now - 3 * H4:
                continue
            k = k.ffill()
            c, h, l = (k[x].to_numpy() for x in ("c", "h", "l"))
            if len(c) >= 120:                       # market breadth: share of coins above their 4h EMA50, now and 7 days ago
                e50 = pd.Series(c).ewm(span=50, adjust=False).mean().to_numpy()
                above_now.append(c[-1] > e50[-1])
                above_7d.append(c[-43] > e50[-43])
            trail, pos = trail_pos(c, h, l)
            states[s] = {"line": float(trail[-1]), "close": float(c[-1]), "pos": int(pos[-1]),
                         "bar_close": int((k.index[-1] + H4).timestamp()), "close_24h_ago": float(c[-6])}
            for i in range(max(1, len(c) - days * 6 - 1), len(c)):
                if pos[i] != pos[i - 1] and pos[i] != 0:
                    close_t = k.index[i] + H4
                    if close_t < since:
                        continue
                    items.append({"id": f"{s}|{int(close_t.timestamp())}", "symbol": s, "side": "buy" if pos[i] == 1 else "sell",
                                  "time": int(close_t.timestamp()) - 900, "price": float(c[i]), "line": float(trail[i]),
                                  "now": float(c[-1] / c[i] - 1), "current": bool(pos[-1] == pos[i]),
                                  "ran24": float(c[i] / c[i - 6] - 1) if i >= 6 else None})
        except Exception as e:
            print(f"atr signals {s}: {e}")
    return {"items": items, "states": states, "built": time.time(), "market": market_state(above_now, above_7d)}


def market_state(above_now, above_7d):
    """'Weak market': BTC below its 4h EMA50 AND fewer coins above their EMA50 than 7 days ago. In tests (10,829 ATR
    Buys, 2023-26) Buys taken then did clearly worse: +5/-3 26-32% vs 34-38%, 3-day return -2.2% in 2025-26."""
    out = {"breadth": float(np.mean(above_now)) if above_now else None,
           "breadth_7d": float(np.mean(above_7d)) if above_7d else None, "btc_vs_ema50": None, "weak": False}
    try:
        b = radar.load("BTCUSDT").ffill()
        e50 = b["c"].ewm(span=50, adjust=False).mean()
        out["btc_vs_ema50"] = float(b["c"].iloc[-1] / e50.iloc[-1] - 1)
    except Exception as e:
        print(f"market state: {e}")
    if out["breadth"] is not None and out["btc_vs_ema50"] is not None:
        out["weak"] = bool(out["btc_vs_ema50"] < 0 and out["breadth"] <= out["breadth_7d"])
    return out


def scan(symbols, days=60):
    since = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    items = []
    for s in sorted(symbols):
        try:
            items += _one(s, min(since, TEST_START))
        except Exception as e:
            print(f"atr_sm {s}: {e}")
    fin = [x for x in items if x["done"] and pd.Timestamp(x["time"], unit="s", tz="UTC") >= TEST_START]
    rec = {"since": int(TEST_START.timestamp()), "finished": len(fin), "long_now": sum(1 for x in items if x["long_now"])}
    if fin:
        for up, dn in GRID:
            key = f"t{int(up*100)}_{int(dn*100)}"
            dec = [x for x in fin if x[key] in ("hit", "stopped", "neither")]
            rec[key] = float(np.mean([x[key] == "hit" for x in dec])) if dec else None
        rec["best_median"] = float(np.median([x["best_until_sell"] for x in fin]))
    return {"items": items, "record": rec, "built": time.time(), "settings": {"period": PERIOD, "mult": MULT}}
