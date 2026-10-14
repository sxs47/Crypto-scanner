"""
Market tab: total crypto market cap (BTC / altcoins / dominance, CoinGecko free API), sentiment (Fear & Greed from
alternative.me, Bybit funding / open interest / long-short ratio) and a coin's technicals on 1d / 4h / 1h / 15m
(EMA 7/21/50/200, RSI 14, MACD 12/26/9, volume vs its 20-candle average, ATR_SM 5/5 state).
Everything is read live and cached for a minute or more; snapshots of the market cap are appended to
data/market/global.csv so a history builds up over time.
"""

import csv
import math
import time

import numpy as np
import pandas as pd
import requests
import ta

import atrstop
import collector
import exchange
import news
import scanner

TIMEFRAMES = (("1d", "D"), ("4h", "240"), ("1h", "60"), ("15m", "15"))
EMAS = (7, 21, 50, 200)
HISTORY = scanner.DATA_DIR / "market" / "global.csv"
_session = requests.Session()
_session.headers["User-Agent"] = "pump-scanner (personal use)"
_last_saved = [0.0]


def _f(x):
    x = float(x)
    return None if math.isnan(x) or math.isinf(x) else x


CMC = "https://pro-api.coinmarketcap.com"


def _cmc(path):
    key = collector.keys().get("coinmarketcap")
    if not key:
        return None
    r = _session.get(CMC + path, headers={"X-CMC_PRO_API_KEY": key, "Accept": "application/json"}, timeout=15)
    r.raise_for_status()
    return r.json()["data"]


def global_caps():
    """Total, BTC, ETH and altcoin market caps with 24h changes: CoinMarketCap when its key is saved
    (cached 10 min, ~4,300 of the free plan's 10,000 monthly credits), otherwise CoinGecko (cached 5 min)."""
    try:
        if collector.keys().get("coinmarketcap"):
            return news._cached("mkt_global_cmc", 600, _cmc_caps)
    except Exception as e:
        print(f"coinmarketcap: {e} — using CoinGecko")
    return news._cached("mkt_global", 300, _gecko_caps)


def _cmc_caps():
    g = _cmc("/v1/global-metrics/quotes/latest")
    u = g["quote"]["USD"]
    total, alts = u["total_market_cap"], u["altcoin_market_cap"]
    btc, eth = total * g["btc_dominance"] / 100, total * g["eth_dominance"] / 100
    total_y = u.get("total_market_cap_yesterday") or total / (1 + (u.get("total_market_cap_yesterday_percentage_change") or 0) / 100)
    dom_y = g.get("btc_dominance_yesterday") or g["btc_dominance"]
    eth_y = g.get("eth_dominance_yesterday") or g["eth_dominance"]
    btc_y, ethc_y = total_y * dom_y / 100, total_y * eth_y / 100
    out = {"source": "CoinMarketCap", "total": total, "btc": btc, "eth": eth, "alts": alts, "alts_ex_eth": alts - eth,
           "btc_dom": g["btc_dominance"], "eth_dom": g["eth_dominance"], "dom_chg": g["btc_dominance"] - dom_y,
           "total_chg": total / total_y - 1, "btc_chg": btc / btc_y - 1, "eth_chg": eth / ethc_y - 1,
           "alts_chg": (total - btc) / (total_y - btc_y) - 1, "alts_ex_eth_chg": (total - btc - eth) / (total_y - btc_y - ethc_y) - 1,
           "updated": int(time.time())}
    _save(out)
    return out


def _gecko_caps():
    g = _session.get("https://api.coingecko.com/api/v3/global", timeout=15).json()["data"]
    total = g["total_market_cap"]["usd"]
    pct = g["market_cap_percentage"]
    btc, eth = total * pct["btc"] / 100, total * pct["eth"] / 100
    chg = g.get("market_cap_change_percentage_24h_usd") or 0.0
    out = {"total": total, "btc": btc, "eth": eth, "alts": total - btc, "alts_ex_eth": total - btc - eth,
           "btc_dom": pct["btc"], "eth_dom": pct["eth"], "total_chg": chg / 100,
           "updated": g.get("updated_at") or int(time.time())}
    try:   # BTC / ETH 24h change from Bybit
        t = {x["symbol"]: float(x["price24hPcnt"]) for x in exchange.get("/v5/market/tickers", {"category": "linear"})["list"]
             if x["symbol"] in ("BTCUSDT", "ETHUSDT")}
        out.update(btc_chg=t["BTCUSDT"], eth_chg=t["ETHUSDT"])
    except Exception as e:
        print(f"market caps: {e}")
    old = _day_ago()          # altcoin 24h change from our own snapshot of a day ago (once we have one)
    if old:
        out["alts_chg"] = (total - btc) / (old["total"] - old["btc"]) - 1
        out["alts_ex_eth_chg"] = (total - btc - eth) / (old["total"] - old["btc"] - old["eth"]) - 1
        out["total_chg"] = total / old["total"] - 1
        out["dom_chg"] = pct["btc"] - old["btc_dom"]
    out["source"] = "CoinGecko"
    _save(out)
    return out


def _day_ago():
    if not HISTORY.exists():
        return None
    h = pd.read_csv(HISTORY)
    target = time.time() - 86400
    h = h[(h.time - target).abs() <= 3600]
    return None if h.empty else h.iloc[(h.time - target).abs().argmin()].to_dict()


def _save(g):
    if time.time() - _last_saved[0] < 14 * 60:
        return
    _last_saved[0] = time.time()
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    new = not HISTORY.exists()
    with HISTORY.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "total", "btc", "eth", "btc_dom"])
        w.writerow([int(time.time()), round(g["total"]), round(g["btc"]), round(g["eth"]), round(g["btc_dom"], 3)])


def fear_greed():
    """Crypto Fear & Greed index (alternative.me, cached 30 min): today, yesterday, a week and a month ago;
    CoinMarketCap's own index too when its key is saved."""
    out = news._cached("mkt_fng", 1800, _alt_fng)
    try:
        if collector.keys().get("coinmarketcap"):
            out = dict(out, cmc=news._cached("mkt_fng_cmc", 1800, lambda: {k: _cmc("/v3/fear-and-greed/latest")[k] for k in ("value", "value_classification")}))
    except Exception as e:
        print(f"coinmarketcap fear & greed: {e}")
    return out


def _alt_fng():
    d = _session.get("https://api.alternative.me/fng/?limit=31", timeout=15).json()["data"]
    pick = lambda i: {"value": int(d[i]["value"]), "label": d[i]["value_classification"]} if len(d) > i else None
    return {"now": pick(0), "yesterday": pick(1), "week": pick(7), "month": pick(30)}


def derivatives(symbol):
    """Bybit perpetual positioning (cached 1 min): funding, open interest + 24h change, long/short account ratio."""
    def fetch():
        t = exchange.get("/v5/market/tickers", {"category": "linear", "symbol": symbol})["list"][0]
        oi = exchange.get("/v5/market/open-interest", {"category": "linear", "symbol": symbol, "intervalTime": "1h", "limit": 25})["list"]
        ls = exchange.get("/v5/market/account-ratio", {"category": "linear", "symbol": symbol, "period": "1h", "limit": 25})["list"]
        price = float(t["lastPrice"])
        out = {"funding": float(t["fundingRate"]), "next_funding": int(t["nextFundingTime"]) // 1000,
               "oi_usd": float(t["openInterestValue"]), "turnover": float(t["turnover24h"]), "change_24h": float(t["price24hPcnt"]),
               "price": price, "high_24h": float(t["highPrice24h"]), "low_24h": float(t["lowPrice24h"])}
        if len(oi) >= 25:
            out["oi_chg"] = float(oi[0]["openInterest"]) / float(oi[24]["openInterest"]) - 1
        if ls:
            out["long_share"] = float(ls[0]["buyRatio"])
            if len(ls) >= 25:
                out["long_share_24h"] = float(ls[24]["buyRatio"])
        return out
    return news._cached(f"mkt_der_{symbol}", 60, fetch)


def technicals(symbol, iv):
    """One timeframe: values on the live (still open) candle, volume on the last closed one."""
    rows = exchange.get("/v5/market/kline", {"category": "linear", "symbol": symbol, "interval": iv, "limit": 1000})["list"]
    d = pd.DataFrame([r[:7] for r in rows], columns=["t", "o", "h", "l", "c", "v", "q"]).astype(float).iloc[::-1].reset_index(drop=True)
    c, h, l, q = d.c, d.h, d.l, d.q
    price = c.iloc[-1]
    ema = {}
    for n in EMAS:
        v = c.ewm(span=n, adjust=False).mean().iloc[-1]
        ema[str(n)] = {"value": _f(v), "dist": _f(price / v - 1)}
    rsi = ta.momentum.RSIIndicator(c, 14).rsi()
    m = ta.trend.MACD(c)
    vol_ratio = q.iloc[-2] / q.iloc[-22:-2].mean()
    _, pos = atrstop.trail_pos(c.to_numpy(), h.to_numpy(), l.to_numpy())
    j = len(pos) - 1
    while j > 0 and pos[j - 1] == pos[-1]:
        j -= 1
    up = sum(price > e["value"] for e in ema.values()) + (rsi.iloc[-1] > 50) + (m.macd_diff().iloc[-1] > 0) + (pos[-1] == 1)
    return {
        "price": _f(price), "candle_chg": _f(price / c.iloc[-2] - 1), "ema": ema,
        "rsi": _f(rsi.iloc[-1]), "rsi_prev": _f(rsi.iloc[-2]),
        "macd": _f(m.macd().iloc[-1]), "macd_signal": _f(m.macd_signal().iloc[-1]), "macd_hist": _f(m.macd_diff().iloc[-1]),
        "macd_hist_prev": _f(m.macd_diff().iloc[-2]),
        "vol_ratio": _f(vol_ratio), "vol_up": bool(c.iloc[-2] >= d.o.iloc[-2]),
        "atr_pos": int(pos[-1]), "atr_bars": int(len(pos) - 1 - j),
        "score": int(up), "score_max": len(EMAS) + 3,
    }


def overview(symbol="BTCUSDT"):
    def fetch():
        out = {"symbol": symbol, "built": time.time(), "tf": {}}
        for name, fn in (("caps", global_caps), ("fng", fear_greed)):
            try:
                out[name] = fn()
            except Exception as e:
                out[name] = None
                print(f"market {name}: {e}")
        try:
            out["deriv"] = derivatives(symbol)
        except Exception as e:
            out["deriv"] = None
            print(f"market derivatives: {e}")
        for name, iv in TIMEFRAMES:
            try:
                out["tf"][name] = technicals(symbol, iv)
            except Exception as e:
                print(f"market {symbol} {name}: {e}")
        return out
    return news._cached(f"mkt_{symbol}", 45, fetch)
