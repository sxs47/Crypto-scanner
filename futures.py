"""
Futures context from Bybit USDT perpetuals: open interest, funding rate and the
long/short account ratio. Public endpoints, no API key.

Bybit keeps a limited history of open interest and the long/short ratio, so
anything learned from it rests on a short sample.

    python futures.py --download   # cache ~30 days of 1h history for every tracked coin
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

import exchange
import scanner

CACHE_DIR = scanner.DATA_DIR / "futures"


def perpetuals():
    """Every tracked coin is a Bybit perpetual."""
    return set(exchange.symbols())


def _paged(path, params, key, ts_key, start_ms, end_ms, limit):
    """Walk a cursor-paged Bybit history endpoint over [start_ms, end_ms]."""
    out, cursor = [], ""
    while True:
        res = exchange.get(path, {**params, "startTime": start_ms, "endTime": end_ms, "limit": limit,
                                  **({"cursor": cursor} if cursor else {})})
        rows = res.get("list", [])
        out += rows
        cursor = res.get("nextPageCursor") or ""
        if not cursor or not rows:
            break
    return sorted({int(r[ts_key]): r for r in out}.values(), key=lambda r: int(r[ts_key]))


def history(symbol, period="1h", days=29):
    """Frame indexed by UTC time: open_interest (coins), long_short, funding (as of each time)."""
    end = int(time.time() * 1000)
    start = end - days * 86_400_000
    oi_period = {"1h": "1h", "15m": "15min"}[period]
    oi = _paged("/v5/market/open-interest", {"category": "linear", "symbol": symbol, "intervalTime": oi_period},
                "list", "timestamp", start, end, 200)
    ls = _paged("/v5/market/account-ratio", {"category": "linear", "symbol": symbol, "period": oi_period},
                "list", "timestamp", start, end, 500)
    fr = _paged("/v5/market/funding/history", {"category": "linear", "symbol": symbol},
                "list", "fundingRateTimestamp", start, end, 200)
    idx = lambda rows, k: pd.to_datetime([int(r[k]) for r in rows], unit="ms", utc=True)
    df = pd.DataFrame(index=idx(oi, "timestamp"))
    if oi:
        df["open_interest"] = [float(r["openInterest"]) for r in oi]
    if ls:
        ratio = [float(r["buyRatio"]) / max(float(r["sellRatio"]), 1e-9) for r in ls]
        df = df.join(pd.Series(ratio, index=idx(ls, "timestamp"), name="long_short"), how="outer")
    if fr:
        f = pd.Series([float(r["fundingRate"]) for r in fr], index=idx(fr, "fundingRateTimestamp"), name="funding")
        f.index = f.index.floor("min")
        df = df.join(f, how="outer")
    df = df[~df.index.duplicated()].sort_index()
    if "funding" in df:
        df["funding"] = df["funding"].ffill()  # funding settles every 1–8h; carry it forward
    return df


def download_all(workers=4, progress=None):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    syms = sorted(perpetuals())
    done = 0

    def one(sym):
        history(sym).to_csv(CACHE_DIR / f"{sym}.csv")
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
    oi = past["open_interest"].dropna() if "open_interest" in past else pd.Series(dtype=float)

    def chg(hours):
        if len(oi) < 2:
            return None
        ref = oi.loc[: t - pd.Timedelta(hours=hours)]
        return float(oi.iloc[-1] / ref.iloc[-1] - 1) if len(ref) else None

    ls = past["long_short"].dropna() if "long_short" in past else pd.Series(dtype=float)
    fr = past["funding"].dropna() if "funding" in past else pd.Series(dtype=float)
    return {
        "oi_chg_4h": chg(4), "oi_chg_24h": chg(24),
        "funding": float(fr.iloc[-1]) if len(fr) else None,
        "long_short": float(ls.iloc[-1]) if len(ls) else None,
    }


def live_context(symbol):
    """Fresh context for one coin (used by the alert detail panel)."""
    meta = exchange.symbols().get(symbol)
    if not meta:
        return None
    df = history(symbol, period="15m", days=2)
    ctx = context_at(df, df.index[-1]) if len(df) else {}
    tick = exchange.get("/v5/market/tickers", {"category": "linear", "symbol": symbol})["list"][0]
    ctx["oi_value"] = float(tick.get("openInterestValue") or 0)
    ctx["funding_next"] = float(tick.get("fundingRate") or 0)
    ctx["next_funding_time"] = int(tick.get("nextFundingTime") or 0) // 1000
    ctx["funding_interval_h"] = meta.get("funding_interval_h", 8)
    if ctx.get("funding") is None:
        ctx["funding"] = ctx["funding_next"]
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
