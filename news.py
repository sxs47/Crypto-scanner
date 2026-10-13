"""
"Why is it moving?" — free context for a coin, no API keys:
  - news headlines from Google News RSS (last 3 days; personal, non-commercial use)
  - Bybit announcements that mention the coin (listings, delistings, launchpools, campaigns)
  - whether it is trending on CoinGecko right now (the coins people search for most)
The collector also saves CoinGecko's trending list every 15 minutes to data/social/trending_YYYY-MM.csv,
so we can later test whether "trending" predicts anything.
"""

import csv
import re
import threading
import time
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

import requests

import exchange
import scanner

SOCIAL_DIR = scanner.DATA_DIR / "social"
_session = requests.Session()
_session.headers["User-Agent"] = "Mozilla/5.0 (pump-scanner personal dashboard)"
_cache, _lock = {}, threading.Lock()


def _cached(key, ttl, fn):
    with _lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < ttl:
            return hit[1]
    val = fn()
    with _lock:
        _cache[key] = (time.time(), val)
    return val


def coin_names(symbol):
    """('PEPE', 'Pepe') for 1000PEPEUSDT — ticker without the 1000× prefix, plus Bybit's full name."""
    base = re.sub(r"^\d+", "", symbol.removesuffix("USDT")) or symbol.removesuffix("USDT")
    name = (exchange.symbols().get(symbol, {}).get("name") or base)
    name = re.sub(r"^\d+", "", name) or base
    name = re.sub(r"\s+(Token|Coin|Network|Protocol)$", "", name, flags=re.I) or base
    return base, name


# a headline must also sound like crypto news (drops "Magic: The Gathering", "Magic Johnson", …)
CRYPTO_WORDS = re.compile(r"crypto|token|coin|price|surge|soar|rall|jump|pump|drop|plung|crash|slump|%|usdt|bitcoin|btc|"
                          r"binance|bybit|okx|upbit|coinbase|listing|delist|airdrop|whale|chart|bull|bear|altcoin|defi|"
                          r"web3|blockchain|staking|unlock|mainnet|exchange|trader|trading|market", re.I)


def _mentions(text, base, name):
    """The ticker in CAPITALS (or $TICKER) always counts; the full name counts in any case — unless the
    name is just the ticker (MAGIC, CAP…), which is often an everyday word."""
    if re.search(rf"(?<![A-Za-z0-9]){re.escape(base.upper())}(?![A-Za-z0-9])", text):
        return True
    if name.lower() != base.lower() and len(name) >= 3:
        return re.search(rf"(?<![a-z0-9]){re.escape(name.lower())}(?![a-z0-9])", text.lower()) is not None
    return False


# --------------------------------------------------------------------------- #
def trending():
    """{TICKER: rank} from CoinGecko's free trending list (cached 10 min)."""
    def fetch():
        r = _session.get("https://api.coingecko.com/api/v3/search/trending", timeout=15)
        r.raise_for_status()
        return [{"rank": i + 1, "symbol": c["item"]["symbol"].upper(), "name": c["item"]["name"],
                 "mcap_rank": c["item"].get("market_cap_rank")} for i, c in enumerate(r.json().get("coins", []))]
    try:
        return _cached("trending", 600, fetch)
    except Exception:
        return []


def save_trending():
    rows = trending()
    if not rows:
        return 0
    SOCIAL_DIR.mkdir(parents=True, exist_ok=True)
    path = SOCIAL_DIR / f"trending_{time.strftime('%Y-%m', time.gmtime())}.csv"
    new = not path.exists()
    now = int(time.time())
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "rank", "symbol", "name", "mcap_rank"])
        w.writerows([[now, r["rank"], r["symbol"], r["name"], r["mcap_rank"]] for r in rows])
    return len(rows)


def bybit_announcements():
    def fetch():
        out = []
        for page in (1, 2):
            res = exchange.get("/v5/announcements/index", {"locale": "en-US", "limit": 50, "page": page})
            out += [{"title": a["title"], "url": a["url"], "time": int(a.get("publishTime") or a.get("dateTimestamp") or 0) // 1000,
                     "type": (a.get("type") or {}).get("title", "")} for a in res.get("list", [])]
        return out
    try:
        return _cached("bybit_ann", 1800, fetch)
    except Exception:
        return []


def google_news(symbol, days=3):
    base, name = coin_names(symbol)
    terms = f'"{base}"' if name.lower() == base.lower() else f'("{name}" OR "{base}")'
    q = f"{terms} (crypto OR token OR coin OR price) when:{days}d"
    url = f"https://news.google.com/rss/search?q={quote_plus(q)}&hl=en-US&gl=US&ceid=US:en"

    def fetch():
        r = _session.get(url, timeout=15)
        r.raise_for_status()
        items = []
        for it in ET.fromstring(r.content).iter("item"):
            title = (it.findtext("title") or "").strip()
            src = it.find("source")
            source = src.text if src is not None else ""
            if source and title.endswith(f" - {source}"):
                title = title[: -len(source) - 3]
            try:
                ts = int(parsedate_to_datetime(it.findtext("pubDate")).timestamp())
            except Exception:
                ts = 0
            if _mentions(title, base, name) and CRYPTO_WORDS.search(title):
                items.append({"title": title, "url": it.findtext("link"), "source": source, "time": ts})
        items.sort(key=lambda x: -x["time"])
        return items[:10]
    return _cached(f"gnews|{symbol}", 600, fetch)


def why(symbol):
    base, name = coin_names(symbol)
    tr = {t["symbol"]: t for t in trending()}
    ann = [a for a in bybit_announcements() if _mentions(a["title"], base, name)][:5]
    try:
        heads, err = google_news(symbol), None
    except Exception as e:
        heads, err = [], f"news unavailable ({e})"
    return {"symbol": symbol, "ticker": base, "name": name, "headlines": heads, "bybit": ann,
            "trending": tr.get(base), "error": err, "fetched": time.time()}
