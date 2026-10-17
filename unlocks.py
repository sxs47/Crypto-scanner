"""
Upcoming big token unlocks for Bybit coins, from DefiLlama's public emissions file (the same data its website shows).
Only one-time ("cliff") unlocks of at least 10% of the circulating supply within the next 30 days are flagged.

Tested 2023-26 (1,295 cliff unlocks >= 0.5% on 98 Bybit coins, vs the market): the effect is small — about -2.4%
in the last 7 days before unlocks of >= 10% of supply, ~0 after; ATR Buys with a big unlock ahead were slightly
worse. So this is only a note, not a signal. Data is refreshed once a day into data/unlocks/.
"""

import json
import time

import requests

import scanner

STORE = scanner.DATA_DIR / "unlocks"
INDEX_URL = "https://defillama-datasets.llama.fi/emissionsIndex"
COINS_URL = "https://api.coingecko.com/api/v3/coins/list"
MIN_PCT = 0.10
AHEAD_DAYS = 30
_state = {"loaded_at": 0.0, "upcoming": {}}


def _get(url, path, max_age):
    if path.exists() and time.time() - path.stat().st_mtime < max_age:
        return json.loads(path.read_text(encoding="utf-8"))
    r = requests.get(url, timeout=120, headers={"User-Agent": "pump-scanner (personal use)"})
    r.raise_for_status()
    path.write_text(r.text, encoding="utf-8")
    return r.json()


def refresh():
    """Rebuild {bybit symbol: next big cliff unlock} (downloads at most once a day)."""
    import exchange
    STORE.mkdir(parents=True, exist_ok=True)
    data = _get(INDEX_URL, STORE / "emissionsIndex.json", 86400)["data"]
    coins = {c["id"]: c["symbol"].upper() for c in _get(COINS_URL, STORE / "coingecko_coins.json", 7 * 86400)}
    perps = set(exchange.symbols()) | set(_bybit_all())
    now, until = time.time(), time.time() + AHEAD_DAYS * 86400
    upcoming = {}
    for t in data:
        sym = coins.get(t.get("gecko_id") or "")
        circ = t.get("circSupply") or 0
        if not sym or circ <= 0:
            continue
        bybit = next((p for p in (f"{sym}USDT", f"1000{sym}USDT", f"10000{sym}USDT") if p in perps), None)
        if not bybit:
            continue
        by_day = {}
        for e in t.get("events") or []:
            ts = e.get("timestamp")
            if e.get("unlockType") != "cliff" or not ts or not now <= ts <= until:
                continue
            n = sum(x for x in (e.get("noOfTokens") or []) if isinstance(x, (int, float)))
            d = by_day.setdefault(int(ts // 86400), {"time": ts, "tokens": 0.0, "who": set()})
            d["tokens"] += n
            d["who"].add(e.get("category") or "other")
        big = [d for d in by_day.values() if d["tokens"] / circ >= MIN_PCT]
        if big:
            d = min(big, key=lambda x: x["time"])
            upcoming[bybit] = {"time": int(d["time"]), "pct": d["tokens"] / circ, "who": sorted(d["who"])}
    _state.update(loaded_at=time.time(), upcoming=upcoming)
    return upcoming


def _bybit_all():
    try:
        import radar
        return radar.all_perpetuals()
    except Exception:
        return []


def next_big(symbol):
    """The next unlock of >= 10% of supply within 30 days for this coin, or None."""
    u = _state["upcoming"].get(symbol)
    return u if u and u["time"] > time.time() else None
