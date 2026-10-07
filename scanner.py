"""
Binance USDT pump scanner.

1. Downloads 15-minute candles for every trading USDT spot pair on Binance
   for the last N days (default 90) via the public REST API (no API key).
2. Labels every candle from which price rose 30%+ within the next 24 hours.
3. Groups those labeled candles into distinct pump events and prints a summary.

Usage:
    python scanner.py                 # download (incremental) + analyze
    python scanner.py --skip-download # analyze existing CSVs only
    python scanner.py --days 30 --threshold 0.5 --window-hours 12
"""

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

# api.binance.com is geo-blocked in some regions; data-api.binance.vision is
# Binance's public market-data-only mirror and serves the same endpoints.
BASE_URLS = ["https://api.binance.com", "https://data-api.binance.vision"]
INTERVAL = "15m"
INTERVAL_MS = 15 * 60 * 1000
KLINE_LIMIT = 1000
WEIGHT_SOFT_LIMIT = 4800  # Binance allows 6000/min per IP; stay under it

# Stablecoins / fiat-pegged bases never pump 30% and just add noise
EXCLUDED_BASES = {
    "USDC", "FDUSD", "TUSD", "USDP", "DAI", "BUSD", "EUR", "EURI", "AEUR",
    "GBP", "PAX", "USDE", "USD1", "XUSD", "BFUSD", "PYUSD", "RLUSD", "UST",
    "PAXG", "XAUT",  # gold-backed tokens
}
# Binance lists tokenized US stocks/ETFs (AAPLB, NVDAB, SPYB…) as USDT spot pairs.
# They spike at the US market open, not on crypto news, so they're excluded.
# Binance has no explicit flag; they're the pairs in this trading group.
STOCK_TOKEN_GROUP = "TRD_GRP_261"

CSV_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote",
]

DATA_DIR = Path(__file__).parent / "data"
CANDLE_DIR = DATA_DIR / "candles"
EXCLUDED_PATH = DATA_DIR / "excluded_symbols.json"


def excluded_symbols():
    """Stock tokens recorded at the last symbol refresh, plus excluded bases."""
    try:
        return set(json.loads(EXCLUDED_PATH.read_text()))
    except (OSError, ValueError):
        return set()


def candle_paths():
    """CSV files for crypto pairs only (skips stock tokens and excluded bases)."""
    skip = excluded_symbols()
    return [p for p in sorted(CANDLE_DIR.glob("*.csv"))
            if p.stem not in skip and p.stem.removesuffix("USDT") not in EXCLUDED_BASES]

_session = requests.Session()
_weight_lock = threading.Lock()
_base_url = None


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def pick_base_url():
    for url in BASE_URLS:
        try:
            r = _session.get(f"{url}/api/v3/ping", timeout=10)
            if r.status_code == 200:
                return url
        except requests.RequestException:
            pass
    sys.exit("Could not reach any Binance endpoint: " + ", ".join(BASE_URLS))


def api_get(path, params=None, retries=6):
    for attempt in range(retries):
        try:
            r = _session.get(f"{_base_url}{path}", params=params, timeout=30)
        except requests.RequestException as e:
            wait = 2 ** attempt
            print(f"  network error ({e.__class__.__name__}), retrying in {wait}s")
            time.sleep(wait)
            continue

        if r.status_code in (418, 429):
            wait = int(r.headers.get("Retry-After", 60))
            print(f"  rate limited ({r.status_code}), sleeping {wait}s")
            time.sleep(wait)
            continue
        if r.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()

        # Back off before Binance does it for us
        used = int(r.headers.get("X-MBX-USED-WEIGHT-1M", 0))
        if used > WEIGHT_SOFT_LIMIT:
            with _weight_lock:
                secs = 61 - datetime.now().second
                print(f"  weight {used}/6000 used, pausing {secs}s")
                time.sleep(secs)
        return r.json()
    raise RuntimeError(f"GET {path} failed after {retries} attempts")


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def get_usdt_symbols():
    info = api_get("/api/v3/exchangeInfo", {"permissions": "SPOT"})
    usdt = [s for s in info["symbols"]
            if s["quoteAsset"] == "USDT" and s["status"] == "TRADING"
            and s.get("isSpotTradingAllowed", True) and s["baseAsset"] not in EXCLUDED_BASES]
    groups = lambda s: {g for ps in s.get("permissionSets", []) for g in ps}
    stocks = {s["symbol"] for s in usdt if STOCK_TOKEN_GROUP in groups(s)}
    # Remember them so loaders skip any CSVs already downloaded for these pairs
    DATA_DIR.mkdir(exist_ok=True)
    EXCLUDED_PATH.write_text(json.dumps(sorted(excluded_symbols() | stocks), indent=0))
    return sorted(s["symbol"] for s in usdt if s["symbol"] not in stocks)


def download_symbol(symbol, start_ms, end_ms):
    """Fetch closed candles in [start_ms, end_ms), resuming from an existing CSV."""
    path = CANDLE_DIR / f"{symbol}.csv"
    existing = None
    fetch_from = start_ms

    if path.exists():
        existing = pd.read_csv(path)
        if len(existing):
            last_open = pd.Timestamp(existing["open_time"].iloc[-1]).value // 10**6
            first_open = pd.Timestamp(existing["open_time"].iloc[0]).value // 10**6
            if first_open <= start_ms + INTERVAL_MS:
                fetch_from = last_open + INTERVAL_MS
            else:
                existing = None  # file covers a shorter history; refetch fully

    rows = []
    cursor = fetch_from
    while cursor < end_ms:
        batch = api_get("/api/v3/klines", {
            "symbol": symbol, "interval": INTERVAL,
            "startTime": cursor, "endTime": end_ms - 1, "limit": KLINE_LIMIT,
        })
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1][0] + INTERVAL_MS
        if len(batch) < KLINE_LIMIT:
            break

    new = pd.DataFrame([r[:11] for r in rows], columns=CSV_COLUMNS)
    if len(new):
        for col in ("open_time", "close_time"):
            new[col] = pd.to_datetime(new[col], unit="ms", utc=True).dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )

    df = pd.concat([existing, new], ignore_index=True) if existing is not None else new
    if not len(df):
        return symbol, 0, 0

    # Trim to the requested window and drop any duplicates from resumes
    cutoff = pd.to_datetime(start_ms, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S")
    df = df[df["open_time"] >= cutoff].drop_duplicates("open_time").sort_values("open_time")
    df.to_csv(path, index=False)
    return symbol, len(new), len(df)


def download_all(days, workers, progress=None):
    """progress(done, total, symbol, error) is called after each pair, if given."""
    global _base_url
    _base_url = pick_base_url()
    print(f"Using {_base_url}")

    CANDLE_DIR.mkdir(parents=True, exist_ok=True)
    symbols = get_usdt_symbols()
    print(f"Found {len(symbols)} trading USDT pairs")

    # Align to candle boundaries; end at the start of the current (unclosed) candle
    now_ms = int(time.time() * 1000)
    end_ms = now_ms - now_ms % INTERVAL_MS
    start_ms = end_ms - days * 24 * 60 * 60 * 1000

    t0 = time.time()
    done = failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_symbol, s, start_ms, end_ms): s for s in symbols}
        for fut in as_completed(futures):
            sym = futures[fut]
            done += 1
            error = None
            try:
                _, added, total = fut.result()
                print(f"[{done}/{len(symbols)}] {sym}: +{added} new, {total} total")
            except Exception as e:
                failed += 1
                error = str(e)
                print(f"[{done}/{len(symbols)}] {sym}: FAILED ({e})")
            if progress:
                progress(done, len(symbols), sym, error)

    print(f"Download finished in {time.time() - t0:.0f}s ({failed} failed)\n")
    return symbols


# --------------------------------------------------------------------------- #
# Labeling
# --------------------------------------------------------------------------- #
def load_candles(path):
    """
    Read a candle CSV onto a full 15m grid (missing candles become NaN rows) so
    trading halts don't stretch a fixed-length window past its nominal hours.
    """
    df = pd.read_csv(path, usecols=["open_time", "open", "high", "low", "close", "quote_volume"])
    if len(df) < 2:
        return None
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.set_index("open_time")
    full = pd.date_range(df.index[0], df.index[-1], freq="15min")
    df = df.reindex(full)
    df.index.name = "open_time"
    return df


def label_symbol(path, threshold, window_hours):
    df = load_candles(path)
    return None if df is None else label_frame(df, threshold, window_hours)


def label_frame(df, threshold, window_hours):
    """
    For every candle t, entry = close[t] and
        future_max = max(high[t+1 .. t+window])
    The candle is labeled a pump if future_max / entry - 1 >= threshold.
    Returns a copy; the input frame is left untouched.
    """
    df = df.copy()
    n = window_hours * 4
    # Max of the *next* n highs: reverse, rolling max, reverse back, shift by one
    fwd_max = df["high"][::-1].rolling(n, min_periods=1).max()[::-1].shift(-1)
    df["future_max"] = fwd_max
    df["gain"] = fwd_max / df["close"] - 1
    df["label"] = df["gain"] >= threshold
    return df


def find_events(symbol, df, threshold, window_hours):
    """Collapse consecutive labeled candles into distinct pump events."""
    hits = df[df["label"]]
    if hits.empty:
        return []

    gap = pd.Timedelta(hours=window_hours)
    # A new event starts when a labeled candle is > window after the previous one
    group = (hits.index.to_series().diff() > gap).cumsum()

    step = pd.Timedelta(minutes=15)
    events = []
    for _, g in hits.groupby(group):
        # Best entry = the labeled candle with the largest forward gain
        best_t = g["gain"].idxmax()
        entry = df.at[best_t, "close"]
        window = df.loc[best_t + step: best_t + gap]
        peak_t = window["high"].idxmax()

        # Speed: from the best entry, how long until +threshold was first touched
        hit_t = window.index[(window["high"] / entry - 1 >= threshold).argmax()]

        events.append({
            "symbol": symbol,
            "first_signal": g.index[0],
            "hours_to_threshold": round((hit_t - best_t).total_seconds() / 3600, 2),
            "entry_time": best_t,
            "entry_price": entry,
            "peak_time": peak_t,
            "peak_price": window["high"].max(),
            "max_gain_pct": round(g["gain"].max() * 100, 2),
            # Same window measured on closes — filters out single-candle wicks
            "max_close_gain_pct": round((window["close"].max() / entry - 1) * 100, 2),
            "hours_to_peak": round((peak_t - best_t).total_seconds() / 3600, 2),
            "labeled_candles": len(g),
            "quote_vol_24h_before": round(
                df.loc[best_t - gap: best_t, "quote_volume"].sum(), 0
            ),
        })
    return events


def analyze(threshold, window_hours, top):
    files = candle_paths()
    if not files:
        sys.exit(f"No CSVs in {CANDLE_DIR}. Run without --skip-download first.")

    labeled_rows, events = [], []
    total_candles = 0
    date_min = date_max = None
    for path in files:
        symbol = path.stem
        df = label_symbol(path, threshold, window_hours)
        if df is None:
            continue
        total_candles += df["close"].notna().sum()
        date_min = min(date_min or df.index[0], df.index[0])
        date_max = max(date_max or df.index[-1], df.index[-1])

        hits = df[df["label"]]
        if len(hits):
            labeled_rows.append(
                hits[["close", "future_max", "gain"]].assign(symbol=symbol).reset_index()
            )
            events.extend(find_events(symbol, df, threshold, window_hours))

    DATA_DIR.mkdir(exist_ok=True)
    labeled = (
        pd.concat(labeled_rows, ignore_index=True)
        if labeled_rows else pd.DataFrame(columns=["open_time", "close", "future_max", "gain", "symbol"])
    )
    labeled = labeled[["symbol", "open_time", "close", "future_max", "gain"]]
    labeled["gain_pct"] = (labeled.pop("gain") * 100).round(2)
    labeled.to_csv(DATA_DIR / "labeled_candles.csv", index=False)

    ev = pd.DataFrame(events)
    if len(ev):
        ev = ev.sort_values("max_gain_pct", ascending=False)
    ev.to_csv(DATA_DIR / "pump_events.csv", index=False)

    print_summary(ev, labeled, len(files), total_candles, date_min, date_max,
                  threshold, window_hours, top)


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def bar(count, biggest, width=50):
    return "█" * max(1 if count else 0, round(count / max(biggest, 1) * width))


def print_summary(ev, labeled, n_symbols, n_candles, dmin, dmax, threshold, hours, top):
    line = "=" * 78
    print(line)
    print(f"PUMP SUMMARY  —  {threshold:.0%}+ gain within {hours}h  (entry = 15m close)")
    print(line)
    print(f"Period:            {dmin:%Y-%m-%d %H:%M} → {dmax:%Y-%m-%d %H:%M} UTC")
    print(f"Pairs scanned:     {n_symbols}")
    print(f"Candles scanned:   {n_candles:,}")
    print(f"Labeled candles:   {len(labeled):,}  ({len(labeled) / max(n_candles, 1):.3%} of all)")
    print(f"Distinct events:   {len(ev)}")
    if ev.empty:
        print(line)
        return
    n_coins = ev["symbol"].nunique()
    print(f"Coins with ≥1 pump: {n_coins}  ({n_coins / n_symbols:.1%} of pairs)")

    g = ev["max_gain_pct"]
    print(f"\nMax gain per event: median {g.median():.1f}%, mean {g.mean():.1f}%, max {g.max():.1f}%")
    buckets = pd.cut(g, [threshold * 100, 50, 75, 100, 200, float("inf")], right=False)
    print("\nGain distribution:")
    counts = buckets.value_counts(sort=False)
    for interval, count in counts.items():
        lo, hi = interval.left, interval.right
        label = f"{lo:.0f}%+" if hi == float("inf") else f"{lo:.0f}–{hi:.0f}%"
        print(f"  {label:>10}: {count:4d}  {bar(count, counts.max())}")

    wick = ev["max_close_gain_pct"] < threshold * 100
    print(f"\nWick-only events (threshold hit on highs, never on a 15m close): "
          f"{wick.sum()} ({wick.mean():.0%})")

    h = ev["hours_to_threshold"]
    print(f"Speed — hours from best entry (local low) to first +{threshold:.0%} touch: "
          f"median {h.median():.1f}h; {(h <= 1).mean():.0%} within 1h, "
          f"{(h <= 4).mean():.0%} within 4h, {(h > 12).mean():.0%} took >12h")

    print(f"\nTop {top} events by gain:")
    cols = ["symbol", "entry_time", "entry_price", "peak_price", "max_gain_pct",
            "max_close_gain_pct", "hours_to_peak"]
    t = ev[cols].head(top).copy()
    t["entry_time"] = t["entry_time"].dt.strftime("%Y-%m-%d %H:%M")
    t["entry_price"] = t["entry_price"].map(lambda x: f"{x:.6g}")
    t["peak_price"] = t["peak_price"].map(lambda x: f"{x:.6g}")
    print(t.to_string(index=False))

    print(f"\nCoins with the most pump events:")
    per_coin = (ev.groupby("symbol")
                  .agg(events=("max_gain_pct", "size"), best_gain_pct=("max_gain_pct", "max"))
                  .sort_values(["events", "best_gain_pct"], ascending=False)
                  .head(top))
    print(per_coin.to_string())

    print("\nEvents per week (by entry time):")
    weekly = ev.set_index("entry_time").resample("W-MON", label="left", closed="left").size()
    for wk, c in weekly.items():
        print(f"  {wk:%Y-%m-%d}: {c:3d}  {bar(c, weekly.max())}")

    print("\nHour of day (UTC) of first qualifying candle:")
    hours_hist = ev["first_signal"].dt.hour.value_counts().reindex(range(24), fill_value=0)
    for hr, c in hours_hist.items():
        print(f"  {hr:02d}:00 {c:3d}  {bar(c, hours_hist.max())}")

    print(line)
    print(f"Saved: {DATA_DIR / 'labeled_candles.csv'}  (every labeled candle)")
    print(f"       {DATA_DIR / 'pump_events.csv'}  (one row per event)")
    print(line)


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--days", type=int, default=90, help="history to download (default 90)")
    p.add_argument("--threshold", type=float, default=0.30, help="gain threshold (default 0.30 = 30%%)")
    p.add_argument("--window-hours", type=int, default=24, help="look-ahead window (default 24)")
    p.add_argument("--workers", type=int, default=8, help="parallel download threads")
    p.add_argument("--top", type=int, default=20, help="rows in top-N tables")
    p.add_argument("--skip-download", action="store_true", help="only analyze existing CSVs")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252

    if not args.skip_download:
        download_all(args.days, args.workers)
    analyze(args.threshold, args.window_hours, args.top)


if __name__ == "__main__":
    main()
