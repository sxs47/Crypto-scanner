"""
Bybit spot coins that are NOT listed on Binance (public API, no key).

Same CSV format as the Binance candles, so the same indicators, setups and
live alerts run on them. Bybit klines carry no trade-count or taker-buy data;
those columns are left empty (none of the alert rules use them).

Excluded: stablecoins/gold (scanner.EXCLUDED_BASES), tokenized stocks
(symbolType "xstocks"), and coins trading under MIN_VOLUME a day.
Coins Bybit flags as risky (stTag, "adventure" zone) are kept but tagged.

    python bybit.py --download     # 90 days of 15m candles for the Bybit-only list
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

BASE = "https://api.bybit.com"
DATA_DIR = scanner.DATA_DIR / "bybit"
CANDLE_DIR = DATA_DIR / "candles"
META_PATH = DATA_DIR / "symbols.json"
MIN_VOLUME = 250_000          # USDT traded per day on Bybit
INTERVAL_MS = 15 * 60 * 1000
EXTRA_STABLES = {"USDC", "RLUSD", "USD1", "USDE", "USDT0", "USDQ", "USDR", "FDUSD", "PYUSD", "DAI", "USDD", "XAUT", "PAXG", "EURC"}
# Wrapped / staked versions of other coins just track their price
PEGGED = {"STETH", "WSTETH", "METH", "CMETH", "WBETH", "BBSOL", "WBTC", "CBBTC", "BTCB", "JITOSOL", "MSOL", "BNSOL"}

_session = requests.Session()
_lock = threading.Lock()
_last = [0.0]
MIN_GAP = 0.02  # Bybit allows ~600 requests / 5s per IP; stay far below


def get(path, params=None, retries=5):
    for attempt in range(retries):
        with _lock:
            wait = _last[0] + MIN_GAP - time.time()
            if wait > 0:
                time.sleep(wait)
            _last[0] = time.time()
        try:
            r = _session.get(BASE + path, params=params, timeout=20)
            data = r.json()
        except (requests.RequestException, ValueError):
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 429 or data.get("retCode") == 10006:  # rate limited
            time.sleep(2 + 2 ** attempt)
            continue
        if r.status_code >= 500 or data.get("retCode") in (10000, 10016) or "internal" in str(data.get("retMsg", "")).lower():
            time.sleep(2 ** attempt)  # transient server-side error
            continue
        if data.get("retCode") != 0:
            raise RuntimeError(f"Bybit {path}: {data.get('retMsg')}")
        return data["result"]
    raise RuntimeError(f"Bybit GET {path} failed")


def refresh_symbols():
    """Bybit-only USDT spot coins worth scanning; saved with their risk tags."""
    rows, cursor = [], ""
    while True:
        res = get("/v5/market/instruments-info", {"category": "spot", "limit": 1000, "cursor": cursor})
        rows += res["list"]
        cursor = res.get("nextPageCursor") or ""
        if not cursor:
            break
    tickers = {t["symbol"]: float(t.get("turnover24h") or 0) for t in get("/v5/market/tickers", {"category": "spot"})["list"]}
    binance = {p.stem for p in scanner.candle_paths()} | scanner.excluded_symbols()
    out = {}
    for x in rows:
        base, sym = x["baseCoin"], x["symbol"]
        if (x["quoteCoin"] != "USDT" or x["status"] != "Trading" or sym in binance
                or base in scanner.EXCLUDED_BASES or base in EXTRA_STABLES or base in PEGGED
                or x.get("symbolType") == "xstocks" or tickers.get(sym, 0) < MIN_VOLUME):
            continue
        tags = []
        if x.get("stTag") == "1":
            tags.append("risk warning")
        if x.get("symbolType") == "adventure":
            tags.append("adventure zone")
        out[sym] = {"volume_24h": tickers.get(sym, 0), "tags": tags}
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    META_PATH.write_text(json.dumps(out, indent=1))
    return out


def symbols():
    try:
        return json.loads(META_PATH.read_text())
    except (OSError, ValueError):
        return {}


def candle_paths():
    keep = symbols()
    return [p for p in sorted(CANDLE_DIR.glob("*.csv")) if p.stem in keep]


def to_binance_rows(rows):
    """Bybit kline rows -> Binance-shaped rows (so live.append_rows can store them)."""
    nan = float("nan")
    return [[int(r[0]), r[1], r[2], r[3], r[4], r[5], int(r[0]) + INTERVAL_MS - 1, r[6], nan, nan, nan] for r in rows]


def fetch_klines(symbol, start_ms, end_ms):
    """Closed 15m candles with open time in [start_ms, end_ms), oldest first."""
    out = []
    cursor_end = end_ms - 1
    while cursor_end >= start_ms:
        res = get("/v5/market/kline", {"category": "spot", "symbol": symbol, "interval": "15",
                                       "start": start_ms, "end": cursor_end, "limit": 1000})
        batch = res["list"]  # newest first
        if not batch:
            break
        out.extend(batch)
        oldest = int(batch[-1][0])
        if len(batch) < 1000 or oldest <= start_ms:
            break
        cursor_end = oldest - 1
    out = sorted({int(r[0]): r for r in out if start_ms <= int(r[0]) < end_ms}.values(), key=lambda r: int(r[0]))
    return to_binance_rows(out)


def download_symbol(symbol, days=90):
    now = int(time.time() * 1000)
    end = now - now % INTERVAL_MS
    start = end - days * 86_400_000
    path = CANDLE_DIR / f"{symbol}.csv"
    if path.exists():
        last = pd.read_csv(path, usecols=["open_time"])["open_time"]
        if len(last):
            start = max(start, int(pd.Timestamp(last.iloc[-1], tz="UTC").timestamp() * 1000) + INTERVAL_MS)
    rows = fetch_klines(symbol, start, end)
    if rows:
        raw = pd.DataFrame(rows, columns=scanner.CSV_COLUMNS)
        for col in ("open_time", "close_time"):
            raw[col] = pd.to_datetime(raw[col], unit="ms", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")
        raw.to_csv(path, mode="a", header=not path.exists(), index=False)
    return symbol, len(rows)


def download_all(progress=None, workers=8):
    CANDLE_DIR.mkdir(parents=True, exist_ok=True)
    syms = sorted(refresh_symbols())
    done = 0

    def safe(sym):
        try:
            return download_symbol(sym)
        except Exception as e:  # one bad coin shouldn't stop the rest
            print(f"\nBybit {sym} failed: {e}")
            return sym, 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for sym, n in pool.map(safe, syms):
            done += 1
            if progress:
                progress(done, len(syms), sym)
    return len(syms)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.download:
        t0 = time.time()
        n = download_all(progress=lambda d, n, s: print(f"\r{d}/{n} {s:<16}", end="", flush=True))
        print(f"\nBybit: {n} coins downloaded in {time.time() - t0:.0f}s")
    for sym, m in sorted(symbols().items(), key=lambda kv: -kv[1]["volume_24h"]):
        print(f"  {sym:<14} ${m['volume_24h'] / 1e6:6.2f}M/day  {', '.join(m['tags'])}")


if __name__ == "__main__":
    main()
