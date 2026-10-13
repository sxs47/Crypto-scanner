"""
Background data collection (runs inside app.py, or on its own: python collector.py).

1. Solana memecoin trades, with the trader's wallet address, from GeckoTerminal's
   free public API (no key, ~30 calls/min). A watchlist of trending and new pools
   is refreshed every few minutes; each pool's latest trades are polled in turn.
   → data/onchain/solana/trades_YYYY-MM-DD.csv.gz (gzip), pools_YYYY-MM-DD.csv
   Only pools created in the last 30 days, and trades of at least $20, to keep the files small.
   Busy pools can trade more than the 300 trades the API returns between polls,
   so very active pools have gaps.

4. Every liquidation on Bybit USDT perpetuals, live from Bybit's public websocket (allLiquidation).
   side = "long" means a long position was force-closed (a forced SELL), "short" = a forced BUY.
   → data/liquidations/YYYY-MM-DD.csv.gz   (time_ms, symbol, side, size, price, usd)

3. CoinGecko's trending list (the coins people search for most) every 15 minutes
   → data/social/trending_YYYY-MM.csv   (see news.py)

2. Bybit futures sentiment every 15 minutes for every tracked coin: open interest,
   funding rate and the long/short account ratio, building our own history beyond
   the ~30 days Bybit keeps.
   → data/sentiment/YYYY-MM.csv

API keys for later wallet tools are stored in data/keys.json (not in git):
    python app.py --set-key helius
    python app.py --set-key etherscan
"""

import csv
import getpass
import gzip
import json
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import requests
import websocket

import exchange
import news
import scanner

GT = "https://api.geckoterminal.com/api/v2"
ONCHAIN_DIR = scanner.DATA_DIR / "onchain" / "solana"
SENT_DIR = scanner.DATA_DIR / "sentiment"
KEYS_PATH = scanner.DATA_DIR / "keys.json"
GT_GAP = 2.2                  # seconds between GeckoTerminal calls (~27/min)
WATCH_MAX = 30                # pools polled in rotation
REFRESH_POOLS_S = 300         # rebuild the watchlist every 5 minutes
MIN_VOLUME_24H = 50_000       # ignore pools with less trading
MAX_POOL_AGE_DAYS = 30        # memecoin pumps happen in young pools
MIN_TRADE_USD = 20            # smaller trades are mostly bots and dust

status = {"solana": {"pools": 0, "trades_today": 0, "last": None, "error": None},
          "sentiment": {"coins": 0, "last": None, "error": None},
          "trending": {"coins": 0, "last": None, "error": None},
          "liquidations": {"today": 0, "usd_today": 0.0, "coins": 0, "last": None, "error": None}, "running": False}
_session = requests.Session()
_session.headers["accept"] = "application/json"
_gt_lock = threading.Lock()
_gt_last = [0.0]


# --------------------------------------------------------------------------- #
# Keys
# --------------------------------------------------------------------------- #
def keys():
    try:
        return json.loads(KEYS_PATH.read_text())
    except (OSError, ValueError):
        return {}


def set_key_cli(name):
    name = name.lower()
    if name not in ("helius", "etherscan"):
        raise SystemExit("Key name must be 'helius' or 'etherscan'.")
    val = getpass.getpass(f"Paste your {name} API key (it won't be shown): ").strip()
    if not val:
        raise SystemExit("Nothing entered; no change.")
    k = keys()
    k[name] = val
    scanner.DATA_DIR.mkdir(exist_ok=True)
    KEYS_PATH.write_text(json.dumps(k, indent=1))
    print(f"Saved the {name} key to {KEYS_PATH} (private, not in git).")


# --------------------------------------------------------------------------- #
# Solana memecoin trades (GeckoTerminal)
# --------------------------------------------------------------------------- #
def gt_get(path, params=None):
    for attempt in range(4):
        with _gt_lock:
            wait = _gt_last[0] + GT_GAP - time.time()
            if wait > 0:
                time.sleep(wait)
            _gt_last[0] = time.time()
        try:
            r = _session.get(GT + path, params=params, timeout=20)
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 429:
            time.sleep(30)
            continue
        if r.status_code >= 500:
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"GeckoTerminal {path} failed")


def _append(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    opener = gzip.open if path.suffix == ".gz" else open  # gzip appends as extra members; readers handle that
    with opener(path, "at", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerows(rows)


def refresh_pools():
    """Trending + new Solana pools with enough trading → {pool_address: info}."""
    pools = {}
    for path in ("/networks/solana/trending_pools", "/networks/solana/new_pools"):
        for p in gt_get(path, {"include": "base_token"})["data"]:
            a = p["attributes"]
            vol = float((a.get("volume_usd") or {}).get("h24") or 0)
            if vol < MIN_VOLUME_24H:
                continue
            created = a.get("pool_created_at")
            if created:
                try:
                    age_days = (time.time() - time.mktime(time.strptime(created[:19], "%Y-%m-%dT%H:%M:%S"))) / 86400
                except ValueError:
                    age_days = 0
                if age_days > MAX_POOL_AGE_DAYS:
                    continue
            base_id = p["relationships"]["base_token"]["data"]["id"]  # "solana_<mint>"
            pools[a["address"]] = {
                "pool": a["address"], "name": a["name"], "token": base_id.split("_", 1)[-1],
                "price_usd": float(a.get("base_token_price_usd") or 0), "volume_24h": vol,
                "liquidity": float(a.get("reserve_in_usd") or 0), "fdv": float(a.get("fdv_usd") or 0),
                "created": a.get("pool_created_at"), "source": "trending" if "trending" in path else "new",
            }
    ranked = sorted(pools.values(), key=lambda x: -x["volume_24h"])[:WATCH_MAX]
    now = int(time.time())
    _append(ONCHAIN_DIR / f"pools_{time.strftime('%Y-%m-%d', time.gmtime())}.csv",
            ["time", "pool", "token", "name", "price_usd", "volume_24h", "liquidity", "fdv", "created", "source"],
            [[now, p["pool"], p["token"], p["name"], p["price_usd"], p["volume_24h"], p["liquidity"], p["fdv"], p["created"], p["source"]]
             for p in ranked])
    return {p["pool"]: p for p in ranked}


def solana_loop(stop):
    seen = deque(maxlen=200_000)
    seen_set = set()
    watch, last_refresh, i = {}, 0, 0
    today, count = None, 0
    while not stop.is_set():
        try:
            if time.time() - last_refresh > REFRESH_POOLS_S or not watch:
                watch = refresh_pools() or watch
                last_refresh = time.time()
                status["solana"]["pools"] = len(watch)
            if not watch:
                time.sleep(30)
                continue
            pool = list(watch.values())[i % len(watch)]
            i += 1
            data = gt_get(f"/networks/solana/pools/{pool['pool']}/trades")["data"]
            rows = []
            for t in data:
                a = t["attributes"]
                if float(a["volume_in_usd"] or 0) < MIN_TRADE_USD:
                    continue
                key = (a["tx_hash"], a["kind"], a["tx_from_address"])
                if key in seen_set:
                    continue
                if len(seen) == seen.maxlen:
                    seen_set.discard(seen[0])
                seen.append(key)
                seen_set.add(key)
                price = a["price_to_in_usd"] if a["kind"] == "buy" else a["price_from_in_usd"]
                rows.append([a["block_timestamp"], pool["pool"], pool["token"], a["kind"],
                             round(float(a["volume_in_usd"] or 0), 2), f"{float(price or 0):.6g}", a["tx_from_address"], a["tx_hash"]])
            day = time.strftime("%Y-%m-%d", time.gmtime())
            if day != today:
                today, count = day, 0
            if rows:
                rows.sort(key=lambda r: r[0])
                _append(ONCHAIN_DIR / f"trades_{day}.csv.gz",
                        ["time", "pool", "token", "kind", "usd", "price_usd", "wallet", "tx"], rows)
                count += len(rows)
            status["solana"].update(trades_today=count, last=int(time.time()), error=None)
        except Exception as e:
            status["solana"]["error"] = f"{time.strftime('%H:%M')}: {e}"
            time.sleep(20)


# --------------------------------------------------------------------------- #
# Bybit sentiment (open interest, funding, long/short)
# --------------------------------------------------------------------------- #
def sentiment_snapshot():
    syms = exchange.symbols()
    tick = {t["symbol"]: t for t in exchange.get("/v5/market/tickers", {"category": "linear"})["list"]}

    def ls(sym):
        try:
            r = exchange.get("/v5/market/account-ratio", {"category": "linear", "symbol": sym, "period": "15min", "limit": 1})["list"]
            return sym, (float(r[0]["buyRatio"]) / max(float(r[0]["sellRatio"]), 1e-9)) if r else None
        except Exception:
            return sym, None

    with ThreadPoolExecutor(max_workers=6) as pool:
        ratios = dict(pool.map(ls, syms))
    now = int(time.time())
    rows = []
    for sym in syms:
        t = tick.get(sym)
        if not t:
            continue
        rows.append([now, sym, t.get("lastPrice"), t.get("openInterest"), t.get("openInterestValue"),
                     t.get("fundingRate"), ratios.get(sym), t.get("turnover24h")])
    _append(SENT_DIR / f"{time.strftime('%Y-%m', time.gmtime())}.csv",
            ["time", "symbol", "price", "open_interest", "oi_value", "funding", "long_short", "turnover_24h"], rows)
    status["sentiment"].update(coins=len(rows), last=now, error=None)


def sentiment_loop(stop):
    while not stop.is_set():
        try:
            sentiment_snapshot()
        except Exception as e:
            status["sentiment"]["error"] = f"{time.strftime('%H:%M')}: {e}"
        # next run ~1 minute after the next 15-minute mark
        nxt = (time.time() // 900 + 1) * 900 + 60
        stop.wait(max(5, nxt - time.time()))


def trending_loop(stop):
    while not stop.is_set():
        try:
            n = news.save_trending()
            status["trending"].update(coins=n, last=int(time.time()), error=None if n else "no data")
        except Exception as e:
            status["trending"]["error"] = f"{time.strftime('%H:%M')}: {e}"
        stop.wait(900)


LIQ_DIR = scanner.DATA_DIR / "liquidations"
WS_URL = "wss://stream.bybit.com/v5/public/linear"


def _flush_liqs(buf):
    if not buf:
        return
    day = time.strftime("%Y-%m-%d", time.gmtime())
    _append(LIQ_DIR / f"{day}.csv.gz", ["time_ms", "symbol", "side", "size", "price", "usd"], buf)


def liquidation_loop(stop):
    """Record every liquidation (public websocket). Reconnects on any error."""
    backoff = 5
    while not stop.is_set():
        ws = None
        try:
            syms = sorted(exchange.symbols())
            ws = websocket.create_connection(WS_URL, timeout=20)
            for i in range(0, len(syms), 10):                     # Bybit takes up to 10 topics per request
                ws.send(json.dumps({"op": "subscribe", "args": [f"allLiquidation.{s}" for s in syms[i:i + 10]]}))
            status["liquidations"].update(coins=len(syms), error=None)
            backoff = 5
            buf, last_flush, last_ping, today = [], time.time(), time.time(), None
            while not stop.is_set():
                if time.time() - last_ping > 20:
                    ws.send(json.dumps({"op": "ping"}))
                    last_ping = time.time()
                try:
                    msg = json.loads(ws.recv())
                except websocket.WebSocketTimeoutException:
                    continue
                day = time.strftime("%Y-%m-%d", time.gmtime())
                if day != today:
                    _flush_liqs(buf); buf = []
                    today = day
                    status["liquidations"].update(today=0, usd_today=0.0)
                if str(msg.get("topic", "")).startswith("allLiquidation"):
                    for d in msg.get("data", []):
                        size, price = float(d["v"]), float(d["p"])
                        side = "long" if d["S"] == "Buy" else "short"   # Bybit: "Buy" = a long position was liquidated
                        buf.append([int(d["T"]), d["s"], side, size, price, round(size * price, 2)])
                        status["liquidations"]["today"] += 1
                        status["liquidations"]["usd_today"] += size * price
                        status["liquidations"]["last"] = int(time.time())
                if time.time() - last_flush > 60:
                    _flush_liqs(buf); buf = []
                    last_flush = time.time()
            _flush_liqs(buf)
        except Exception as e:
            status["liquidations"]["error"] = f"{time.strftime('%H:%M')}: {e}"
            stop.wait(backoff)
            backoff = min(backoff * 2, 300)
        finally:
            try:
                ws and ws.close()
            except Exception:
                pass


_stop = threading.Event()


def start():
    if status["running"]:
        return
    status["running"] = True
    threading.Thread(target=solana_loop, args=(_stop,), daemon=True).start()
    threading.Thread(target=sentiment_loop, args=(_stop,), daemon=True).start()
    threading.Thread(target=trending_loop, args=(_stop,), daemon=True).start()
    threading.Thread(target=liquidation_loop, args=(_stop,), daemon=True).start()


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    start()
    print("Collecting… (Ctrl+C to stop)")
    while True:
        time.sleep(60)
        print(time.strftime("%H:%M"), json.dumps(status))
