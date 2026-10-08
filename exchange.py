"""
Market data source: Bybit USDT perpetual futures (public API, no key).

Every other module gets its coin list and candles from here.

Coin list = Bybit USDT perpetuals that are trading, crypto only (Bybit's
symbolType "stock", "ETF", "commodity" and "forex" are excluded, as are
stablecoins and gold), with at least MIN_VOLUME traded per day. Coins Bybit
puts in its "innovation" zone (new / higher-risk listings) are kept but tagged.

Candles are returned in the row shape the rest of the code expects:
[open_ms, open, high, low, close, base_volume, close_ms, quote_volume, nan, nan, nan]
(Bybit has no trade-count or taker-buy fields.) Only closed candles are returned.
"""

import json
import threading
import time

import requests

import scanner

BASE = "https://api.bybit.com"
CATEGORY = "linear"
MIN_VOLUME = 250_000          # USDT traded per day
NON_CRYPTO = {"stock", "ETF", "commodity", "forex"}
INTERVALS = {"15m": ("15", 900_000), "1h": ("60", 3_600_000), "4h": ("240", 14_400_000), "1d": ("D", 86_400_000)}
SYMBOLS_PATH = scanner.DATA_DIR / "symbols.json"

_session = requests.Session()
_lock = threading.Lock()
_last = [0.0]
MIN_GAP = 0.02  # Bybit allows ~600 requests / 5 s per IP; this stays far below


def get(path, params=None, retries=6):
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
    raise RuntimeError(f"Bybit GET {path} failed after {retries} attempts")


def refresh_symbols():
    """Fetch the coin list from Bybit, apply the filters, save it with tags. Returns {symbol: meta}."""
    rows, cursor = [], ""
    while True:
        res = get("/v5/market/instruments-info", {"category": CATEGORY, "limit": 1000, "cursor": cursor})
        rows += res["list"]
        cursor = res.get("nextPageCursor") or ""
        if not cursor:
            break
    vol = {t["symbol"]: float(t.get("turnover24h") or 0) for t in get("/v5/market/tickers", {"category": CATEGORY})["list"]}
    out = {}
    for x in rows:
        sym, base = x["symbol"], x["baseCoin"]
        if (x["contractType"] != "LinearPerpetual" or x["quoteCoin"] != "USDT" or x["status"] != "Trading"
                or x.get("symbolType") in NON_CRYPTO or base in scanner.EXCLUDED_BASES or vol.get(sym, 0) < MIN_VOLUME):
            continue
        out[sym] = {
            "volume_24h": vol.get(sym, 0),
            "tags": ["innovation"] if x.get("symbolType") == "innovation" else [],
            "name": x.get("fullName") or base,
            "listed": int(x.get("launchTime") or 0) // 1000,
            "funding_interval_h": int(x.get("fundingInterval") or 480) / 60,
        }
    scanner.DATA_DIR.mkdir(exist_ok=True)
    SYMBOLS_PATH.write_text(json.dumps(out, indent=1))
    return out


def symbols():
    """The saved coin list (refresh_symbols() writes it)."""
    try:
        return json.loads(SYMBOLS_PATH.read_text())
    except (OSError, ValueError):
        return {}


def klines(symbol, interval, start_ms, end_ms):
    """Closed candles with open time in [start_ms, end_ms), oldest first, in the shared row shape."""
    code, step = INTERVALS[interval]
    now = int(time.time() * 1000)
    end_ms = min(end_ms, now - now % step)  # never the candle that's still open
    out, cursor_end = [], end_ms - 1
    while cursor_end >= start_ms:
        batch = get("/v5/market/kline", {"category": CATEGORY, "symbol": symbol, "interval": code,
                                         "start": start_ms, "end": cursor_end, "limit": 1000})["list"]  # newest first
        if not batch:
            break
        out.extend(batch)
        oldest = int(batch[-1][0])
        if len(batch) < 1000 or oldest <= start_ms:
            break
        cursor_end = oldest - 1
    rows = sorted({int(r[0]): r for r in out if start_ms <= int(r[0]) < end_ms}.values(), key=lambda r: int(r[0]))
    nan = float("nan")
    return [[int(r[0]), r[1], r[2], r[3], r[4], r[5], int(r[0]) + step - 1, r[6], nan, nan, nan] for r in rows]


def trade_url(symbol):
    return f"https://www.bybit.com/trade/usdt/{symbol}"
