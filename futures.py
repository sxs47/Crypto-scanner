"""
Binance USDT-M perpetual futures context: open interest, funding rate and the
global long/short account ratio. Free public endpoints, no API key.

Binance only serves the last ~30 days of open-interest and long/short history,
so anything learned from it rests on a short sample.

    python futures.py --download   # cache 30 days of 1h history for every coin with a perpetual
"""

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

import scanner

BASE = "https://fapi.binance.com"
CACHE_DIR = scanner.DATA_DIR / "futures"
_session = requests.Session()
_throttle = threading.Lock()
_last_call = [0.0]
MIN_GAP = 0.32  # seconds between calls: ~940 per 5 min, under the 1000/5min data-endpoint limit
_perps = None


def get(path, params=None, retries=5):
    for attempt in range(retries):
        with _throttle:
            wait = _last_call[0] + MIN_GAP - time.time()
            if wait > 0:
                time.sleep(wait)
            _last_call[0] = time.time()
        try:
            r = _session.get(BASE + path, params=params, timeout=20)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if r.status_code in (418, 429):
            time.sleep(int(r.headers.get("Retry-After", 30)))
            continue
        if r.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"GET {path} failed")


def perpetuals():
    global _perps
    if _perps is None:
        info = get("/fapi/v1/exchangeInfo")
        _perps = {s["symbol"] for s in info["symbols"]
                  if s["contractType"] == "PERPETUAL" and s["quoteAsset"] == "USDT" and s["status"] == "TRADING"}
    return _perps


def _paged(path, symbol, period, days, limit=500):
    """openInterestHist / globalLongShortAccountRatio, walking forward in pages."""
    step = {"15m": 900, "1h": 3600}[period] * 1000
    end = int(time.time() * 1000)
    start = end - days * 86_400_000
    out = []
    while start < end:
        batch = get(path, {"symbol": symbol, "period": period, "limit": limit,
                           "startTime": start, "endTime": min(end, start + step * limit)})
        if not batch:
            start += step * limit
            continue
        out.extend(batch)
        start = int(batch[-1]["timestamp"]) + step
    return out


def history(symbol, period="1h", days=29):
    """Wide frame indexed by UTC time: oi_value, long_short, plus funding (as of each time)."""
    oi = _paged("/futures/data/openInterestHist", symbol, period, days)
    ls = _paged("/futures/data/globalLongShortAccountRatio", symbol, period, days)
    fr = get("/fapi/v1/fundingRate", {"symbol": symbol, "limit": 1000,
                                      "startTime": int((time.time() - days * 86400) * 1000)})
    idx = lambda rows, key="timestamp": pd.to_datetime([int(r[key]) for r in rows], unit="ms", utc=True)
    df = pd.DataFrame(index=idx(oi))
    if oi:
        df["oi_value"] = [float(r["sumOpenInterestValue"]) for r in oi]
    if ls:
        df = df.join(pd.Series([float(r["longShortRatio"]) for r in ls], index=idx(ls), name="long_short"), how="outer")
    if fr:
        f = pd.Series([float(r["fundingRate"]) for r in fr], index=idx(fr, "fundingTime"), name="funding")
        f.index = f.index.floor("min")
        df = df.join(f, how="outer")
    df = df[~df.index.duplicated()].sort_index()
    if "funding" in df:
        df["funding"] = df["funding"].ffill()  # funding settles every 8h (sometimes 4h/1h); carry it forward
    return df


def download_all(workers=4, progress=None):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    spot = {p.stem for p in scanner.candle_paths()}
    syms = sorted(spot & perpetuals())
    done = 0

    def one(sym):
        df = history(sym)
        df.to_csv(CACHE_DIR / f"{sym}.csv")
        return sym

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for sym in pool.map(one, syms):
            done += 1
            if progress:
                progress(done, len(syms), sym)
    return len(syms)


def load(symbol):
    path = CACHE_DIR / f"{symbol}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def context_at(df, t):
    """Futures context known at time t (no look-ahead)."""
    past = df.loc[:t]
    if past.empty:
        return None
    oi = past["oi_value"].dropna() if "oi_value" in past else pd.Series(dtype=float)

    def chg(hours):
        if len(oi) < 2:
            return None
        ref = oi.loc[: t - pd.Timedelta(hours=hours)]
        return float(oi.iloc[-1] / ref.iloc[-1] - 1) if len(ref) else None

    ls = past["long_short"].dropna() if "long_short" in past else pd.Series(dtype=float)
    fr = past["funding"].dropna() if "funding" in past else pd.Series(dtype=float)
    return {
        "oi_value": float(oi.iloc[-1]) if len(oi) else None,
        "oi_chg_4h": chg(4), "oi_chg_24h": chg(24),
        "funding": float(fr.iloc[-1]) if len(fr) else None,
        "long_short": float(ls.iloc[-1]) if len(ls) else None,
    }


def live_context(symbol):
    """Fresh context for one coin (used by the alert detail panel)."""
    if symbol not in perpetuals():
        return None
    df = history(symbol, period="15m", days=2)
    ctx = context_at(df, df.index[-1])
    prem = get("/fapi/v1/premiumIndex", {"symbol": symbol})
    ctx["funding_next"] = float(prem.get("lastFundingRate", 0))
    ctx["next_funding_time"] = int(prem.get("nextFundingTime", 0)) // 1000
    # Funding settles every 8h for most coins but 4h or 1h for some; measure it
    recent = get("/fapi/v1/fundingRate", {"symbol": symbol, "limit": 4})
    gaps = [int(b["fundingTime"]) - int(a["fundingTime"]) for a, b in zip(recent, recent[1:])]
    ctx["funding_interval_h"] = round(min(gaps) / 3_600_000) if gaps else 8
    ctx["series"] = {
        "t": [int(x.timestamp()) for x in df.index],
        "oi": [None if pd.isna(v) else float(v) for v in df.get("oi_value", pd.Series(index=df.index))],
        "ls": [None if pd.isna(v) else float(v) for v in df.get("long_short", pd.Series(index=df.index))],
    }
    return ctx


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.download:
        t0 = time.time()
        n = download_all(progress=lambda d, n, s: print(f"\r{d}/{n} {s:<16}", end="", flush=True))
        print(f"\nCached futures history for {n} coins in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
