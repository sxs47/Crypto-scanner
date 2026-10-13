"""
▲ Run picks — which coins have the best chance to run UP before they fall?

A model scores every coin at each 4h close using ~173 TradingView-style indicators (python `ta`, on 4h and daily
candles), our volume features, and Bybit POSITIONING: open interest, long/short account ratio and funding.
It predicts, for the next 3 days, whether +15% or −15% comes first; a pick = the top 0.5% of
"chance of a run − chance of a crash".

Research (Bybit 4h, 2023–2026, trained on one period and tested on the other): long picks reached +10% before −5%
in 41–46% of cases and +20% before −10% in 41–47% (break-even 33%); with a −10% stop the typical pick rose
+15–19% first (random moments ~+7.5%). Positioning mattered most — the long/short ratio works contrarian
(crowded shorts → more runs up). Paper-tracked here; not advice.

    python runpicks.py --train      update the stores, build the dataset, train, save models/runpicks_model.pkl
"""

import json
import pickle
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import atrstop
import exchange
import radar
import scanner

warnings.filterwarnings("ignore")
POS_DIR = scanner.DATA_DIR / "positioning_4h"
FUND_DIR = scanner.DATA_DIR / "funding"
MODEL_PATH = Path(__file__).parent / "models" / "runpicks_model.pkl"
REPORT_PATH = Path(__file__).parent / "models" / "runpicks_report.json"
START = pd.Timestamp("2023-01-01", tz="UTC")
TRAIN_END = pd.Timestamp("2026-07-01", tz="UTC")
TOP = 0.005
UP, DN, HORIZON = 0.15, 0.15, 18          # label: ±15% first within 3 days
GRID = ((0.10, 0.05), (0.20, 0.10))       # tracked outcomes: target before stop
MAXBARS = 42                               # follow each pick for 7 days
TAIL = 1300                                # 4h candles used when scoring live (≈ 7 months)
FEE = 0.0011
RESEARCH = {"p10_5": "41–46%", "p20_10": "41–47%", "best10": "+15% to +19%", "random_p10_5": "29%", "random_p20_10": "18–19%",
            "random_best10": "+7.5%", "breakeven": "33%"}
PRICE_LEVEL = ("bbm", "bbh", "bbl", "kcc", "kch", "kcl", "dcl", "dch", "dcm", "sma_fast", "sma_slow", "ema_fast",
               "ema_slow", "ichimoku_conv", "ichimoku_base", "ichimoku_a", "ichimoku_b", "visual", "vwap")
CUMUL = ("volume_adi", "volume_obv", "volume_vpt", "volume_nvi")
DROP = ("others_cr", "psar_up_indicator", "psar_down_indicator")


# --------------------------------------------------------------------------- #
# Positioning store (open interest + long/short, 4h) and funding store
# --------------------------------------------------------------------------- #
def _paged(path, params, key, limit, start_ms):
    out, end = [], int(time.time() * 1000)
    while end > start_ms:
        b = exchange.get(path, {**params, "startTime": start_ms, "endTime": end, "limit": limit})["list"]
        if not b:
            break
        out += b
        oldest = min(int(x[key]) for x in b)
        if len(b) < limit or oldest <= start_ms:
            break
        end = oldest - 1
    return out


def _last_t(path):
    try:
        return int(pd.read_csv(path, usecols=["t"]).iloc[-1, 0])
    except Exception:
        return None


def _update_one(sym):
    pp, fp = POS_DIR / f"{sym}.csv", FUND_DIR / f"{sym}.csv"
    start = int(START.timestamp() * 1000)
    lt = _last_t(pp)
    s = (lt + 1) if lt else start
    oi = _paged("/v5/market/open-interest", {"category": "linear", "symbol": sym, "intervalTime": "4h"}, "timestamp", 200, s)
    ls = _paged("/v5/market/account-ratio", {"category": "linear", "symbol": sym, "period": "4h"}, "timestamp", 500, s)
    a = pd.DataFrame([(int(x["timestamp"]), float(x["openInterest"])) for x in oi], columns=["t", "oi"])
    b = pd.DataFrame([(int(x["timestamp"]), float(x["buyRatio"]), float(x["sellRatio"])) for x in ls], columns=["t", "buy", "sell"])
    d = a.merge(b, on="t", how="outer").sort_values("t")
    d = d[d["t"] > (lt or 0)].drop_duplicates("t")
    if len(d):
        d.to_csv(pp, mode="a", header=not pp.exists(), index=False)
    lf = _last_t(fp)
    fr = _paged("/v5/market/funding/history", {"category": "linear", "symbol": sym}, "fundingRateTimestamp", 200, (lf + 1) if lf else start)
    f = pd.DataFrame([(int(x["fundingRateTimestamp"]), float(x["fundingRate"])) for x in fr], columns=["t", "rate"])
    f = f[f["t"] > (lf or 0)].drop_duplicates("t").sort_values("t")
    if len(f):
        f.to_csv(fp, mode="a", header=not fp.exists(), index=False)
    return len(d)


def update_store(symbols, workers=6):
    POS_DIR.mkdir(parents=True, exist_ok=True)
    FUND_DIR.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(workers) as pool:
        return sum(n or 0 for n in pool.map(lambda s: radar._safe(_update_one, s), symbols))


# --------------------------------------------------------------------------- #
# Features (only data known at each 4h close)
# --------------------------------------------------------------------------- #
def _indicators(df):
    import ta
    x = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "q": "volume"})[["open", "high", "low", "close", "volume"]]
    f = ta.add_all_ta_features(x.copy(), "open", "high", "low", "close", "volume", fillna=False)
    f = f.drop(columns=["open", "high", "low", "close", "volume"])
    c = x["close"]
    out = {}
    for col in f.columns:
        if any(d in col for d in DROP) or "psar_up" in col or "psar_down" in col:
            continue
        s = f[col]
        if col in CUMUL:
            s = s.diff(6) / (x["volume"].rolling(6).sum() + 1) if col != "volume_nvi" else s.pct_change(6)
        elif any(k in col for k in PRICE_LEVEL):
            s = s / c - 1
        elif col == "volatility_atr":
            s = s / c
        out[col] = s
    if "trend_psar_up" in f:
        out["trend_psar_dist"] = f["trend_psar_up"].fillna(f["trend_psar_down"]) / c - 1
    return pd.DataFrame(out, index=df.index)


def features(sym, tail=None):
    """Feature frame for one coin on its 4h grid (index = candle open time; values known at that candle's close)."""
    k = radar.load(sym)
    pp, fp = POS_DIR / f"{sym}.csv", FUND_DIR / f"{sym}.csv"
    if k is None or not pp.exists() or not fp.exists():
        return None, None
    k = k.ffill()
    if tail:
        k = k.iloc[-tail:]
    if len(k) < 400:
        return None, None
    f4 = _indicators(k).add_prefix("4h_")
    dd = k.resample("1D").agg({"o": "first", "h": "max", "l": "min", "c": "last", "q": "sum"}).dropna()
    f1 = _indicators(dd).add_prefix("1d_")
    f1.index = f1.index + pd.Timedelta(days=1)
    f1 = f1.reindex(k.index, method="ffill")
    c, h, l, q = k["c"], k["h"], k["l"], k["q"]
    v24 = q.rolling(6).sum()
    normal = pd.concat([v24.shift(6 * j) for j in range(1, 31)], axis=1).median(axis=1, skipna=False)
    extra = pd.DataFrame({
        "vol24_x": v24 / normal, "ret_24h": c / c.shift(6) - 1, "ret_7d": c / c.shift(42) - 1, "ret_30d": c / c.shift(180) - 1,
        "from_30d_high": c / h.rolling(180).max() - 1, "from_30d_low": c / l.rolling(180).min() - 1,
        "size_log10": np.log10(normal.clip(lower=1)), "hour": k.index.hour.astype(float), "weekday": k.index.weekday.astype(float),
    }, index=k.index)
    # positioning
    close_t = k.index + pd.Timedelta(hours=4)
    d = pd.read_csv(pp)
    d.index = pd.to_datetime(d["t"], unit="ms", utc=True)
    d = d[~d.index.duplicated(keep="last")].sort_index()
    asof = lambda s: pd.Series(s.dropna().reindex(close_t, method="ffill", tolerance=pd.Timedelta(hours=4)).to_numpy(), index=k.index)
    oi = asof(d["oi"])
    ls = asof(d["buy"] / d["sell"].replace(0, np.nan))
    pos = pd.DataFrame({
        "pos_oi_chg_4h": oi / oi.shift(1) - 1, "pos_oi_chg_24h": oi / oi.shift(6) - 1, "pos_oi_chg_7d": oi / oi.shift(42) - 1,
        "pos_oi_vs_30d": oi / oi.rolling(180, min_periods=60).median(),
        "pos_oi_to_volume": oi * c / v24.replace(0, np.nan),
        "pos_ls": ls, "pos_ls_chg_24h": ls - ls.shift(6),
        "pos_ls_z30d": (ls - ls.rolling(180, min_periods=60).mean()) / ls.rolling(180, min_periods=60).std(),
    }, index=k.index)
    pos["pos_price_x_oi"] = np.sign(c / c.shift(6) - 1) * np.sign(pos["pos_oi_chg_24h"])
    fr = pd.read_csv(fp)
    fr.index = pd.to_datetime(fr["t"], unit="ms", utc=True)
    fr = fr["rate"][~fr.index.duplicated(keep="last")].sort_index()
    cum = fr.cumsum()
    cat = lambda tt: cum.reindex(tt, method="ffill").to_numpy()
    pos["pos_fund_last"] = fr.reindex(close_t, method="ffill").to_numpy()
    pos["pos_fund_24h"] = cat(close_t) - cat(close_t - pd.Timedelta(hours=24))
    pos["pos_fund_7d"] = cat(close_t) - cat(close_t - pd.Timedelta(days=7))
    X = pd.concat([f4, f1, extra, pos], axis=1).replace([np.inf, -np.inf], np.nan).astype("float32")
    X["liquid"] = (v24 >= 1_000_000).to_numpy()
    return X, k


def _label(k):
    h, l, c = k["h"].to_numpy(), k["l"].to_numpy(), k["c"].to_numpy()
    n = len(c)
    lab = np.full(n, np.nan)
    for i in range(n - HORIZON):
        e, res = c[i], 0
        for j in range(i + 1, i + HORIZON + 1):
            if l[j] <= e * (1 - DN):
                res = -1; break
            if h[j] >= e * (1 + UP):
                res = 1; break
        lab[i] = res
    return lab


def _train_rows(sym):
    try:
        X, k = features(sym)
    except Exception:
        return None
    if X is None:
        return None
    X["label"] = _label(k)
    X = X.iloc[200:]
    X = X[X["liquid"] & X["label"].notna() & X["vol24_x"].notna() & X["pos_ls"].notna()].drop(columns="liquid")
    rng = np.random.default_rng(len(X))
    X = X[(X["label"] != 0) | (rng.random(len(X)) < 0.10)].copy()
    X["weight"] = np.where(X["label"] != 0, 1.0, 10.0).astype("float32")
    return X


def train():
    from sklearn.ensemble import HistGradientBoostingClassifier
    syms = radar.all_perpetuals()
    print(f"Updating 4h candles and positioning for {len(syms)} perpetuals…", flush=True)
    radar.update_store(syms + ["BTCUSDT"])
    update_store(syms)
    with ProcessPoolExecutor(8) as pool:
        parts = [x for x in pool.map(_train_rows, syms, chunksize=4) if x is not None and len(x)]
    A = pd.concat(parts)
    A = A[A.index < TRAIN_END]
    feats = [c for c in A.columns if c not in ("label", "weight")]
    m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=300,
                                       l2_regularization=1.0, random_state=1)
    m.fit(A[feats], A["label"].astype(int), sample_weight=A["weight"])
    cls = list(m.classes_)
    recent = A[A.index >= TRAIN_END - pd.Timedelta(days=180)]
    P = m.predict_proba(recent[feats])
    score = P[:, cls.index(1)] - P[:, cls.index(-1)]
    cutoff = radar.wquantile(score, recent["weight"].to_numpy(), 1 - TOP)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as fh:
        pickle.dump({"model": m, "feats": feats, "cutoff": cutoff, "classes": cls, "train_end": TRAIN_END.isoformat()}, fh)
    rep = {"trained": time.time(), "train_end": TRAIN_END.isoformat(), "rows": len(A), "coins": len(parts),
           "features": len(feats), "cutoff": cutoff, "research": RESEARCH}
    REPORT_PATH.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


# --------------------------------------------------------------------------- #
# Live scan
# --------------------------------------------------------------------------- #
_model = {"mtime": None, "obj": None}


def model():
    if not MODEL_PATH.exists():
        return None
    mt = MODEL_PATH.stat().st_mtime
    if _model["mtime"] != mt:
        with open(MODEL_PATH, "rb") as fh:
            _model.update(obj=pickle.load(fh), mtime=mt)
    return _model["obj"]


def _outcome(k, i, e):
    """Track the pick on 4h candles: per target/stop pair → 'hit' / 'stopped' / 'open'; best gain before a −5% / −10% stop."""
    h, l, c = k["h"].to_numpy(), k["l"].to_numpy(), k["c"].to_numpy()
    end = min(len(c), i + 1 + MAXBARS)
    res = {}
    for up, dn in GRID:
        st = "open"
        for j in range(i + 1, end):
            if l[j] <= e * (1 - dn):
                st = "stopped"; break
            if h[j] >= e * (1 + up):
                st = "hit"; break
        if st == "open" and i + MAXBARS < len(c):
            st = "neither"
        res[f"t{int(up*100)}_{int(dn*100)}"] = st
    for stop in (0.05, 0.10):
        best, stopped = 0.0, False
        for j in range(i + 1, end):
            if l[j] <= e * (1 - stop):
                stopped = True; break
            best = max(best, h[j] / e - 1)
        res[f"best_{int(stop*100)}"] = float(best)
        res[f"stopped_{int(stop*100)}"] = stopped
    res["done"] = bool(i + MAXBARS < len(c) or res["stopped_10"])
    res["now"] = float(c[-1] / e - 1)
    return res


def _scan_one(args):
    sym, m = args
    try:
        X, k = features(sym, tail=TAIL)
    except Exception:
        return []
    if X is None:
        return []
    X = X[(X.index >= pd.Timestamp(m["train_end"])) & X["liquid"] & X["pos_ls"].notna() & X["vol24_x"].notna()]
    if X.empty:
        return []
    P = m["model"].predict_proba(X[m["feats"]])
    cls = m["classes"]
    kc = k.ffill()
    _, apos = atrstop.trail_pos(kc["c"].to_numpy(), kc["h"].to_numpy(), kc["l"].to_numpy())
    X = X.assign(score=P[:, cls.index(1)] - P[:, cls.index(-1)], p_up=P[:, cls.index(1)], p_dn=P[:, cls.index(-1)])
    out, last = [], None
    for t, r in X[X["score"] >= m["cutoff"]].iterrows():
        if last is not None and t - last < pd.Timedelta(hours=24):
            continue
        last = t
        i = k.index.get_loc(t)
        e = float(k["c"].iloc[i])
        close_t = t + pd.Timedelta(hours=4)
        why = [f"L/S {r['pos_ls']:.2f}" + (" (crowded shorts)" if r["pos_ls_z30d"] <= -1 else " (crowded longs)" if r["pos_ls_z30d"] >= 1 else ""),
               f"funding 7d {r['pos_fund_7d'] * 100:+.2f}%", f"30d {r['ret_30d'] * 100:+.0f}%", f"RSI(1d) {r['1d_momentum_rsi']:.0f}"]
        j = i
        while j > 0 and apos[j - 1] == 1 and apos[i] == 1:
            j -= 1
        atr_long = bool(apos[i] == 1)
        out.append({"id": f"{sym}|{int(close_t.timestamp())}", "symbol": sym, "time": int(close_t.timestamp()) - 900, "price": e,
                    "atr_long": atr_long, "atr_buy_hours": (i - j) * 4 if atr_long else None,
                    "score": float(r["score"]), "p_up": float(r["p_up"]), "p_dn": float(r["p_dn"]), "reasons": why,
                    **_outcome(k, i, e)})
    return out


def scan(symbols):
    m = model()
    if m is None:
        return None
    with ThreadPoolExecutor(4) as pool:
        picks = [p for x in pool.map(_scan_one, [(s, m) for s in sorted(symbols)]) for p in x]
    fin = [p for p in picks if p["done"]]
    rec = {"n": len(picks), "finished": len(fin), "since": int(pd.Timestamp(m["train_end"]).timestamp())}
    if fin:
        for up, dn in GRID:
            key = f"t{int(up*100)}_{int(dn*100)}"
            dec = [p for p in fin if p[key] in ("hit", "stopped", "neither")]
            rec[key] = float(np.mean([p[key] == "hit" for p in dec])) if dec else None
        rec["best10_median"] = float(np.median([p["best_10"] for p in fin]))
        for flag, name in ((True, "with_atr"), (False, "without_atr")):
            g = [p for p in fin if p.get("atr_long") is flag]
            dec = [p for p in g if p["t10_5"] in ("hit", "stopped", "neither")]
            rec[name] = {"n": len(g), "t10_5": float(np.mean([p["t10_5"] == "hit" for p in dec])) if dec else None,
                         "best10": float(np.median([p["best_10"] for p in g])) if g else None}
        rec["best5_median"] = float(np.median([p["best_5"] for p in fin]))
    return {"picks": picks, "record": rec, "built": time.time(),
            "report": json.loads(REPORT_PATH.read_text()) if REPORT_PATH.exists() else None}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if "--train" in sys.argv:
        train()
    else:
        print(__doc__)
