"""
Pre-pump watch: coins whose volume and candle size have jumped while the price
hasn't moved yet — the build-up seen before most +30% pumps.

Rule (chosen on 17 Jul–11 Sep 2026, checked once on 11 Sep–11 Oct):
    last 4h average volume   ≥ 8× normal   (normal = 7-day median, ending 48h earlier)
    last 4h average candle   ≥ 3× normal
    price change last 4h     between −3% and +8%
    price change last 24h    below +15%      (hasn't pumped already)
    24h turnover             ≥ $250K
One flag per coin per 24h. On the unseen month 5.1% of flags touched +30% within
24h vs 1.6% for any coin at any time (3.2×); most flags still don't pump.
"Squeeze setup" = Bybit long/short accounts ratio ≤ 1.3 (more accounts short),
which hit +30% in 8% of flags in that month (small sample).

"Volume before" = the coin's 24h volume ending 4h before the flag, × its normal daily
volume (median of the previous 30 days). Flags with ≥6× ("volatile": the coin was
already exploding) hit +30% within 48h about 2.5× as often as calm ones, but also
dropped −15% about 4× as often — a bigger move, not a better direction.
"""

import time

import numpy as np
import pandas as pd

import exchange
import futures
import scanner

VOL_X, RNG_X = 8.0, 3.0
RET4_MIN, RET4_MAX, RET24_MAX = -0.03, 0.08, 0.15
MIN_TURNOVER = 250_000
SQUEEZE_LS = 1.3
COOLDOWN_S = 24 * 3600
TAIL = 3100           # candles needed: 30-day normal daily volume for "volume before"
VOLATILE_X = 6.0      # volume before ≥ this × normal → "volatile"
FEE = 0.0011          # Bybit taker, both sides


def features(df):
    c, qv = df["close"].ffill(), df["quote_volume"].fillna(0)
    rng = (df["high"] - df["low"]) / c
    base_v = qv.rolling(672, min_periods=300).median().shift(192)
    base_r = rng.rolling(672, min_periods=300).median().shift(192)
    return pd.DataFrame({
        "vol_x": qv.rolling(16).mean() / base_v,
        "rng_x": rng.rolling(16).mean() / base_r,
        "ret4": c / c.shift(16) - 1,
        "ret24": c / c.shift(96) - 1,
        "turnover24": qv.rolling(96).sum(),
    }, index=df.index)


def flags(symbol, df, since=None, last=None):
    """Flags on candles after `since` (UTC Timestamp); `last` = time of this coin's previous flag (unix s)."""
    f = features(df)
    v24 = df["quote_volume"].fillna(0).rolling(96, min_periods=96).sum().to_numpy()
    m = ((f["vol_x"] >= VOL_X) & (f["rng_x"] >= RNG_X) & (f["ret4"] > RET4_MIN) & (f["ret4"] < RET4_MAX)
         & (f["ret24"] < RET24_MAX) & (f["turnover24"] >= MIN_TURNOVER) & df["close"].notna())
    out = []
    for t in f.index[m.to_numpy()]:
        ts = int(t.timestamp())
        if last is not None and ts - last < COOLDOWN_S:
            continue
        last = ts
        if since is not None and t <= since:
            continue
        r = f.loc[t]
        i = df.index.get_loc(t)
        vb = None
        if i >= 96 * 31:
            normal = np.nanmedian(v24[i - 96 * np.arange(1, 31)])
            if normal > 0 and not np.isnan(v24[i - 16]):
                vb = float(v24[i - 16] / normal)
        out.append({"id": f"{symbol}|{ts}", "symbol": symbol, "time": ts, "price": float(df.at[t, "close"]),
                    "vol_x": float(r["vol_x"]), "rng_x": float(r["rng_x"]), "ret4": float(r["ret4"]),
                    "ret24": float(r["ret24"]), "turnover24": float(r["turnover24"]), "ls": None, "vol_before": vb})
    return out


# --------------------------------------------------------------------------- #
# Long/short accounts ratio
# --------------------------------------------------------------------------- #
_ls_hist = {"key": None, "data": {}}


def _ls_history():
    """symbol -> Series of the long/short ratio from the futures cache and the collector's 15-min snapshots."""
    files = sorted((scanner.DATA_DIR / "sentiment").glob("*.csv"))
    key = tuple((p.name, p.stat().st_mtime) for p in files) + (len(list(futures.CACHE_DIR.glob("*.csv"))),)
    if key == _ls_hist["key"]:
        return _ls_hist["data"]
    parts = {}
    for p in futures.CACHE_DIR.glob("*.csv"):
        fu = futures.load(p.stem)
        if fu is not None and "long_short" in fu:
            parts.setdefault(p.stem, []).append(fu["long_short"].dropna())
    for p in files:
        s = pd.read_csv(p, usecols=["time", "symbol", "long_short"]).dropna()
        s["time"] = pd.to_datetime(s["time"], unit="s", utc=True)
        for sym, g in s.groupby("symbol"):
            parts.setdefault(sym, []).append(g.set_index("time")["long_short"])
    data = {}
    for sym, ss in parts.items():
        x = pd.concat(ss).sort_index()
        data[sym] = x[~x.index.duplicated(keep="last")]
    _ls_hist.update(key=key, data=data)
    return data


def ls_at(symbol, ts, live=False):
    """Ratio known at unix time `ts` (≤2h old), or a fresh one from Bybit for a live flag."""
    if live:
        try:
            r = exchange.get("/v5/market/account-ratio", {"category": "linear", "symbol": symbol,
                                                          "period": "15min", "limit": 1})["list"]
            if r:
                return float(r[0]["buyRatio"]) / max(float(r[0]["sellRatio"]), 1e-9)
        except Exception:
            pass
    s = _ls_history().get(symbol)
    if s is None:
        return None
    t = pd.Timestamp(ts + 900, unit="s", tz="UTC")   # known at the candle's close
    past = s.loc[:t]
    if past.empty or t - past.index[-1] > pd.Timedelta(hours=2):
        return None
    return float(past.iloc[-1])


def fill_ls(items):
    for f in items:
        if f["ls"] is None:
            f["ls"] = ls_at(f["symbol"], f["time"])
    return items


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
def outcome(df, flag):
    t = pd.Timestamp(flag["time"], unit="s", tz="UTC")
    entry = flag["price"]
    w = df.loc[t + pd.Timedelta(minutes=15): t + pd.Timedelta(hours=24)].dropna(subset=["close"])
    last = df["close"].dropna()
    res = {"now": float(last.iloc[-1] / entry - 1), "hours": (last.index[-1] - t).total_seconds() / 3600,
           "peak": None, "trough": None, "done": False, "ret_24h": None}
    if w.empty:
        return res
    res.update(peak=float(w["high"].max() / entry - 1), trough=float(w["low"].min() / entry - 1),
               done=len(w) >= 96)
    if res["done"]:
        res["ret_24h"] = float(w["close"].iloc[-1] / entry - 1)
    return res


def _summary(rows):
    n = len(rows)
    if not n:
        return {"n": 0}
    pk = np.array([r["peak"] or 0 for r in rows])
    tr = np.array([r["trough"] or 0 for r in rows])
    r24 = np.array([r["ret_24h"] for r in rows])
    return {"n": n, "hit15": float((pk >= 0.15).mean()), "hit30": float((pk >= 0.30).mean()),
            "drop15": float((tr <= -0.15).mean()), "drop20": float((tr <= -0.20).mean()), "avg_24h": float(r24.mean() - FEE),
            "median_24h": float(np.median(r24) - FEE)}


def track_record(items, frames, days=30):
    """How finished flags from the last `days` played out, overall and by long/short ratio."""
    cutoff = time.time() - days * 86400
    done = []
    for f in items:
        if f["time"] < cutoff or f["symbol"] not in frames:
            continue
        o = outcome(frames[f["symbol"]], f)
        if o["done"]:
            done.append({**f, **o})
    known = [r for r in done if r["ls"] is not None]
    first = min((r["time"] for r in done), default=None)
    return {"days": days, "all": _summary(done),
            "per_day": len(done) / max(1, (time.time() - (first or time.time())) / 86400) if done else 0,
            "squeeze": _summary([r for r in known if r["ls"] <= SQUEEZE_LS]),
            "other": _summary([r for r in known if r["ls"] > SQUEEZE_LS]),
            "volatile": _summary([r for r in done if (r.get("vol_before") or 0) >= VOLATILE_X]),
            "calm": _summary([r for r in done if r.get("vol_before") is not None and r["vol_before"] < VOLATILE_X])}


def base_rates(frames, days=30):
    """Any coin at any 15m candle (24h turnover ≥ $250K): how often did it touch +15% / +30% within 24h?"""
    start = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days + 1)
    hit15 = hit30 = n = 0
    for df in frames.values():
        d = df.loc[start - pd.Timedelta(days=1):]
        if len(d) < 200:
            continue
        lab = scanner.label_frame(d[["open", "high", "low", "close", "quote_volume"]], 0.30, 24)
        ok = (d["quote_volume"].fillna(0).rolling(96).sum() >= MIN_TURNOVER) & lab["gain"].notna()
        ok &= d.index >= start
        ok.iloc[-96:] = False                      # outcome not finished yet
        g = lab["gain"][ok]
        n += len(g)
        hit15 += int((g >= 0.15).sum())
        hit30 += int((g >= 0.30).sum())
    return {"n": n, "hit15": hit15 / n if n else None, "hit30": hit30 / n if n else None, "at": time.time()}


RULE = {"vol_x": VOL_X, "rng_x": RNG_X, "ret4": [RET4_MIN, RET4_MAX], "ret24_max": RET24_MAX,
        "min_turnover": MIN_TURNOVER, "squeeze_ls": SQUEEZE_LS, "volatile_x": VOLATILE_X}
