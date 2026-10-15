"""
Spot-led vs futures-led: for coins that trade on both Bybit spot and the perpetual, the spot share of the last 24h
turnover compared with its own 30-day normal. Below 0.8x normal = "futures-led" (leveraged traders pushing while spot
buyers stay away). Tested on 281 coins, 4h, 2023-26: futures-led ATR_SM Buys, +10% rallies and volume spikes did
worse in both halves (e.g. ATR Buy 3-day return -0.5% / -1.8% vs +1.2% / -0.5% normal). Spot-led was not better
than normal, so this is only a warning.

Spot 4h candles (turnover) are kept for the last ~45 days in data/spot_4h/.
"""

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import exchange
import radar
import scanner

STORE = scanner.DATA_DIR / "spot_4h"
KEEP_DAYS = 45
FUTURES_LED = 0.8
SPOT_LED = 1.25
H4_MS = 14_400_000
_spot = {"at": 0.0, "symbols": set()}


def spot_symbols():
    """Bybit spot USDT pairs that are trading (cached 6 h)."""
    if time.time() - _spot["at"] > 6 * 3600:
        rows, cur = [], ""
        while True:
            r = exchange.get("/v5/market/instruments-info", {"category": "spot", "limit": 1000, "cursor": cur})
            rows += r["list"]
            cur = r.get("nextPageCursor") or ""
            if not cur:
                break
        _spot.update(at=time.time(), symbols={x["symbol"] for x in rows if x["quoteCoin"] == "USDT" and x["status"] == "Trading"})
    return _spot["symbols"]


def _update_one(sym):
    path = STORE / f"{sym}.csv"
    now = int(time.time() * 1000)
    start = now - KEEP_DAYS * 86_400_000
    old = None
    if path.exists():
        old = pd.read_csv(path)
        if len(old):
            start = max(start, int(old.t.iloc[-1]) + H4_MS)
    end = now - now % H4_MS                     # closed candles only
    if start >= end:
        return 0
    rows = exchange.get("/v5/market/kline", {"category": "spot", "symbol": sym, "interval": "240",
                                             "start": start, "end": end - 1, "limit": 1000})["list"]
    new = pd.DataFrame([[int(r[0]), float(r[6])] for r in rows if int(r[0]) < end], columns=["t", "q"])
    d = pd.concat([old, new]) if old is not None else new
    d = d.drop_duplicates("t").sort_values("t")
    d = d[d.t >= now - KEEP_DAYS * 86_400_000]
    d.to_csv(path, index=False)
    return len(new)


def update_store(perp_symbols):
    STORE.mkdir(parents=True, exist_ok=True)
    syms = sorted(set(perp_symbols) & spot_symbols())
    with ThreadPoolExecutor(6) as pool:
        list(pool.map(lambda s: radar._safe(_update_one, s), syms))
    return syms


def lead(symbols):
    """{symbol: spot share of 24h turnover / its 30-day normal} at the last closed 4h candle."""
    out = {}
    for s in symbols:
        try:
            sp = STORE / f"{s}.csv"
            k = radar.load(s)
            if k is None or not sp.exists():
                continue
            d = pd.read_csv(sp)
            sq = pd.Series(d.q.to_numpy(), index=pd.to_datetime(d.t, unit="ms", utc=True))
            k = k[k.index >= sq.index[0]]
            if len(k) < 120:
                continue
            q24 = k.q.fillna(0).rolling(6).sum()
            s24 = sq.reindex(k.index).fillna(0).rolling(6).sum()
            share = s24 / (s24 + q24)
            normal = share.shift(6).rolling(180, min_periods=90).median()
            r = share.iloc[-1] / normal.iloc[-1]
            if np.isfinite(r) and s24.iloc[-1] > 0:
                out[s] = float(r)
        except Exception as e:
            print(f"spotflow {s}: {e}")
    return out


def label(rel):
    if rel is None:
        return None
    return "futures" if rel < FUTURES_LED else ("spot" if rel > SPOT_LED else "normal")
