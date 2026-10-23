"""
Hot coins watchlist: coins showing the traits that came before +30%-in-24h pumps (study of 629 pumps on 15m
candles, Jul 10 - Oct 20 2026). Five checks, each worth one point:

  1. pumped before   — at least one +30%-within-24h move in the last 30 days (repeat pumpers made 70% of pumps)
  2. wild            — last 24h high/low range >= 10%
  3. volume up       — last 24h volume >= 1.5x its 30-day median
  4. OI rising       — open interest higher than 7 days ago
  5. shorts crowded  — under 60% of Bybit accounts are long (squeeze fuel)

Tested (both halves, Aug 10 - Sep 14 and Sep 15 - Oct 20): all 5 -> 10% reached +30% within 24h (6x a normal
coin), 25% reached +15% — but 16-20% DROPPED 15% (6x normal too). 4 of 5 -> +30% 6-7%, -15% 10-13%.
A watchlist for big moves, not a direction signal: wait for a Buy (e.g. ATR 4h) before entering.
"""

import time

import numpy as np
import pandas as pd

import scanner

POS_DIR = scanner.DATA_DIR / "positioning_4h"
DAY = 96                      # 15m candles per day
CHECKS = {
    "pumped": "Pumped +30% within 24h at least once in the last 30 days",
    "wild": "24h high-to-low range of 10% or more",
    "volume": "Last 24h volume at least 1.5x its 30-day normal",
    "oi": "Open interest higher than 7 days ago (new positions coming in)",
    "shorts": "Under 60% of Bybit accounts are long — shorts crowded, squeeze fuel",
}
_pos_cache = {"t": 0, "data": {}}


def _positioning(symbols):
    """{symbol: (oi change over 7d, long-account share)} from the 4h store; re-read every 10 min."""
    if time.time() - _pos_cache["t"] < 600:
        return _pos_cache["data"]
    out = {}
    for s in symbols:
        p = POS_DIR / f"{s}.csv"
        if not p.exists():
            continue
        try:
            x = pd.read_csv(p).drop_duplicates("t").sort_values("t").tail(60)
        except Exception:
            continue
        if len(x) < 43:
            continue
        oi = x["oi"].to_numpy(dtype=float)
        out[s] = (oi[-1] / oi[-43] - 1 if oi[-43] > 0 else None, float(x["buy"].iloc[-1]))
    _pos_cache.update(t=time.time(), data=out)
    return out


def _pumps_30d(h, c):
    """Number of separate +30%-within-24h moves that started in the last 30 days (fully known, so >= 24h ago)."""
    n = len(c)
    lo = max(0, n - 30 * DAY - DAY)
    fut = pd.Series(h[lo:])[::-1].rolling(DAY, min_periods=1).max()[::-1].shift(-1).to_numpy()
    gain = fut / c[lo:] - 1
    hits = np.flatnonzero(gain[: max(0, len(gain) - DAY)] >= 0.30)
    if not len(hits):
        return 0, None
    events = 1 + int((np.diff(hits) > DAY).sum())
    return events, int(lo + hits[-1])


def board(frames, atr_states=None, min_score=4):
    atr_states = atr_states or {}
    pos = _positioning(sorted(frames))
    out = []
    for sym, df in frames.items():
        d = df.tail(31 * DAY + DAY)
        c = d["close"].ffill().to_numpy(dtype=float)
        if len(c) < 8 * DAY or np.isnan(c[-1]):
            continue
        h, l = d["high"].to_numpy(dtype=float), d["low"].to_numpy(dtype=float)
        qv = d["quote_volume"].fillna(0).to_numpy(dtype=float)
        n_p, last_p = _pumps_30d(h, c)
        rng = np.nanmax(h[-DAY:]) / np.nanmin(l[-DAY:]) - 1
        q24 = pd.Series(qv).rolling(DAY).sum().to_numpy()
        norm = np.nanmedian(q24[-30 * DAY:])
        vol_x = q24[-1] / norm if norm > 0 else None
        oi7, longs = pos.get(sym, (None, None))
        checks = {
            "pumped": n_p > 0,
            "wild": rng >= 0.10,
            "volume": vol_x is not None and vol_x >= 1.5,
            "oi": oi7 is not None and oi7 > 0,
            "shorts": longs is not None and longs < 0.60,
        }
        checks = {k: bool(v) for k, v in checks.items()}
        score = sum(checks.values())
        if score < min_score:
            continue
        st = atr_states.get(sym) or {}
        out.append({
            "symbol": sym, "score": score, "checks": checks,
            "pumps_30d": n_p, "last_pump": int(d.index[last_p].timestamp()) if last_p is not None else None,
            "range24": float(rng), "vol_x": float(vol_x) if vol_x is not None else None,
            "oi_7d": float(oi7) if oi7 is not None else None, "longs": longs,
            "ch1h": float(c[-1] / c[-5] - 1), "ch24h": float(c[-1] / c[-DAY - 1] - 1),
            "atr4h": st.get("pos"), "price": float(c[-1]),
        })
    out.sort(key=lambda x: (-x["score"], -(x["vol_x"] or 0)))
    return out
