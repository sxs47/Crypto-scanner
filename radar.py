"""
Big-move radar: a model that scores, at every 4h candle close, how likely a coin is to touch +30%
within the next 24h — learned from ~5,000 pumps on all Bybit USDT perpetuals since 2023.

What it found: pumps mostly come from coins that are already waking up (off their lows, just above
EMA200, rising volume, often pumped recently). Tested on unseen data (trained 2023–24, tested 2025–26),
its top 0.5% picks touched +30% within 24h ~10% of the time vs ~1% for any coin — but ~19% of them
also fell −15%, and fixed target/stop trades lost after fees. So: "big move likely", not "buy".

    python radar.py --train      download/refresh 4h candles for all perpetuals, train, save the model

The model is trained on data up to TRAIN_END; every pick after that date is out-of-sample, so the
track record shown in the app is honest. Candles: data/candles_4h/<SYMBOL>.csv (t,o,h,l,c,q).
"""

import json
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import exchange
import scanner

STORE = scanner.DATA_DIR / "candles_4h"
MODEL_PATH = Path(__file__).parent / "models" / "radar_model.pkl"
REPORT_PATH = Path(__file__).parent / "models" / "radar_report.json"
START = pd.Timestamp("2023-01-01", tz="UTC")
TRAIN_END = pd.Timestamp("2026-07-01", tz="UTC")
TEST_START = pd.Timestamp("2025-01-01", tz="UTC")      # for the honest test model (trained before this)
TOP = 0.005                                             # flag the top 0.5% of scores
D, THR, WIN = 6, 0.30, 6                                # bars per day, +30%, within 24h
MIN_SIZE_LOG = 5.4                                      # normal daily volume ≥ ~$250K
FEE = 0.0011
FEATS = ["vol24_x", "vol4h_x", "vol_trend_3d", "range4h_x", "ret_4h", "ret_24h", "ret_7d", "ret_30d", "from_30d_high",
         "from_30d_low", "pos_90d_range", "rsi14", "vs_ema200", "ema20_above_50", "squeeze_pct", "btc_24h", "btc_7d",
         "alt_weak", "size_log10", "age_days", "pumps_prev_30d", "hour_utc", "weekday"]
H4 = pd.Timedelta(hours=4)


# --------------------------------------------------------------------------- #
# 4h candle store
# --------------------------------------------------------------------------- #
def all_perpetuals():
    rows, cursor = [], ""
    while True:
        res = exchange.get("/v5/market/instruments-info", {"category": "linear", "limit": 1000, "cursor": cursor})
        rows += res["list"]
        cursor = res.get("nextPageCursor") or ""
        if not cursor:
            break
    return sorted(x["symbol"] for x in rows
                  if x["contractType"] == "LinearPerpetual" and x["quoteCoin"] == "USDT" and x["status"] == "Trading"
                  and x.get("symbolType") not in exchange.NON_CRYPTO and x["baseCoin"] not in scanner.EXCLUDED_BASES)


def _update_one(sym):
    path = STORE / f"{sym}.csv"
    start = START
    if path.exists():
        last = pd.read_csv(path, usecols=["t"]).iloc[-1, 0] if path.stat().st_size > 20 else None
        if last is not None:
            start = pd.Timestamp(int(last), unit="ms", tz="UTC") + H4
    rows = exchange.klines(sym, "4h", int(start.timestamp() * 1000), int(time.time() * 1000))
    if not rows:
        return 0
    df = pd.DataFrame([[r[0], r[1], r[2], r[3], r[4], r[7]] for r in rows], columns=["t", "o", "h", "l", "c", "q"])
    df.to_csv(path, mode="a", header=not path.exists(), index=False)
    return len(df)


def update_store(symbols, workers=6):
    STORE.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(workers) as pool:
        return sum(n for n in pool.map(lambda s: _safe(_update_one, s), symbols) if n)


def _safe(fn, *a):
    try:
        return fn(*a)
    except Exception as e:
        print(f"radar: {a} failed ({e})")
        return 0


def load(sym):
    path = STORE / f"{sym}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path).drop_duplicates("t")
    if len(df) < 2:
        return None
    df.index = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df[["o", "h", "l", "c", "q"]].astype(float).sort_index()
    return df.reindex(pd.date_range(df.index[0], df.index[-1], freq="4h"))


def alt_phase(frames):
    """Daily: 1 when an equal-weight index of liquid altcoins is below its 50-day average (known at day start)."""
    C, T = {}, {}
    for s, df in frames.items():
        if s != "BTCUSDT":
            d = df.resample("1D").agg({"c": "last", "q": "sum"})
            C[s], T[s] = d["c"], d["q"]
    C, T = pd.DataFrame(C), pd.DataFrame(T)
    liquid = (T.rolling(7).mean() >= 1_000_000).shift(1, fill_value=False)
    idx = np.log(C / C.shift()).where(liquid).mean(axis=1).fillna(0).cumsum()
    weak = (idx < idx.rolling(50).mean()).astype(float).where(idx.rolling(50).count() >= 50)
    weak.index = weak.index + pd.Timedelta(days=1)
    return weak


# --------------------------------------------------------------------------- #
# Features (only data up to each bar's close)
# --------------------------------------------------------------------------- #
def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def features(df, btc, phase, listed=None):
    c = df["c"].ffill()
    h, l, q = df["h"].fillna(c), df["l"].fillna(c), df["q"].fillna(0)
    rng = (h - l) / c
    v24 = q.rolling(D).sum()
    normal_day = pd.concat([v24.shift(D * k) for k in range(1, 31)], axis=1).median(axis=1, skipna=False)
    ema20, ema50, ema200 = (c.ewm(span=s, adjust=False).mean() for s in (20, 50, 200))
    bw = 4 * c.rolling(20).std() / c.rolling(20).mean()
    hi30, lo30 = h.rolling(30 * D).max(), l.rolling(30 * D).min()
    hi90, lo90 = h.rolling(90 * D, min_periods=30 * D).max(), l.rolling(90 * D, min_periods=30 * D).min()
    b = btc.reindex(df.index).ffill()
    first = df.index[0]
    if listed:
        age0 = (first - pd.Timestamp(listed, unit="s", tz="UTC")).total_seconds() / 86400
    else:
        age0 = 365 if first <= START else 0                 # coin already listed before our history begins
    f = pd.DataFrame({
        "vol24_x": v24 / normal_day,
        "vol4h_x": q / q.rolling(30 * D, min_periods=10 * D).median().shift(D),
        "vol_trend_3d": q.rolling(3 * D).sum() / (q.rolling(3 * D).sum().shift(3 * D) + 1),
        "range4h_x": rng / rng.rolling(30 * D, min_periods=10 * D).median().shift(D),
        "ret_4h": c / c.shift(1) - 1,
        "ret_24h": c / c.shift(D) - 1,
        "ret_7d": c / c.shift(7 * D) - 1,
        "ret_30d": c / c.shift(30 * D) - 1,
        "from_30d_high": c / hi30 - 1,
        "from_30d_low": c / lo30 - 1,
        "pos_90d_range": (c - lo90) / (hi90 - lo90),
        "rsi14": rsi(c),
        "vs_ema200": c / ema200 - 1,
        "ema20_above_50": (ema20 > ema50).astype(float),
        "squeeze_pct": bw.rolling(30 * D).rank(pct=True),
        "btc_24h": b / b.shift(D) - 1,
        "btc_7d": b / b.shift(7 * D) - 1,
        "alt_weak": phase.reindex(df.index.floor("D")).to_numpy(dtype=float),
        "size_log10": np.log10(normal_day.clip(lower=1)),
        "age_days": np.minimum(365, age0 + np.arange(len(df)) / D),
        "hour_utc": df.index.hour,
        "weekday": df.index.weekday,
    }, index=df.index)
    fmax = h[::-1].rolling(WIN, min_periods=1).max()[::-1].shift(-1)
    f["gain24"] = fmax / c - 1
    f["label"] = (f["gain24"] >= THR).astype(float).where(fmax.notna() & c.notna() & (np.arange(len(f)) < len(f) - WIN))
    # pumps in the previous 30 days — a pump only counts once its 24h window has finished
    lab = f["label"].fillna(0).to_numpy().astype(bool)
    prior, cur = np.zeros(len(f)), None
    groups = []
    for i in np.flatnonzero(lab):
        if cur is None or i - cur[-1] > WIN:
            if cur:
                groups.append(cur)
            cur = [i]
        else:
            cur.append(i)
    if cur:
        groups.append(cur)
    for g in groups:
        k = g[-1] + WIN + 1
        prior[k: k + 30 * D] += 1
    f["pumps_prev_30d"] = prior
    return f


def wquantile(x, w, q):
    """Quantile of x where each value counts w times (the training sample keeps only 15% of normal bars)."""
    o = np.argsort(x)
    cw = np.cumsum(w[o]) / w.sum()
    return float(x[o][min(len(x) - 1, np.searchsorted(cw, q))])


def _all_frames(symbols):
    with ThreadPoolExecutor(8) as pool:
        return {s: df for s, df in zip(symbols, pool.map(load, symbols)) if df is not None and df["c"].notna().sum() > 60 * D}


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    syms = all_perpetuals()
    print(f"Updating 4h candles for {len(syms)} perpetuals…")
    update_store(syms + ["BTCUSDT"])
    frames = _all_frames(sorted(set(syms) | {"BTCUSDT"}))
    btc, phase = frames["BTCUSDT"]["c"].ffill(), alt_phase(frames)
    meta = exchange.symbols()
    rng = np.random.default_rng(1)
    parts = []
    for s, df in frames.items():
        f = features(df, btc, phase, meta.get(s, {}).get("listed"))
        f = f.iloc[31 * D:].dropna(subset=["label", "vol24_x"])
        keep = (f["label"] == 1) | (rng.random(len(f)) < 0.15)       # all positives, 15% of the rest
        f = f[keep].copy()
        f["weight"] = np.where(f["label"] == 1, 1.0, 1 / 0.15)
        parts.append(f)
    M = pd.concat(parts)
    model = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                                   min_samples_leaf=200, random_state=1)
    # 1) honest test: train before TEST_START, test TEST_START..TRAIN_END
    tr, te = M[M.index < TEST_START], M[(M.index >= TEST_START) & (M.index < TRAIN_END)]
    m1 = model().fit(tr[FEATS], tr["label"], sample_weight=tr["weight"])
    s = m1.predict_proba(te[FEATS])[:, 1]
    tsz = te["size_log10"] >= MIN_SIZE_LOG
    cut1 = wquantile(s[tsz.to_numpy()], te["weight"].to_numpy()[tsz.to_numpy()], 1 - TOP)
    pick = tsz.to_numpy() & (s >= cut1)
    w = te["weight"].to_numpy()
    test = {"period": f"{TEST_START:%b %Y} – {TRAIN_END - pd.Timedelta(days=1):%b %Y}",
            "auc": float(roc_auc_score(te["label"], s, sample_weight=w)),
            "hit30_picks": float(np.average(te["label"][pick], weights=w[pick])),
            "hit30_any": float(np.average(te["label"][tsz], weights=w[tsz])),
            "picks_share": TOP}
    # 2) final model on everything before TRAIN_END
    fin = M[M.index < TRAIN_END]
    m2 = model().fit(fin[FEATS], fin["label"], sample_weight=fin["weight"])
    last_half = fin[(fin.index >= TRAIN_END - pd.Timedelta(days=180)) & (fin["size_log10"] >= MIN_SIZE_LOG)]
    cutoff = wquantile(m2.predict_proba(last_half[FEATS])[:, 1], last_half["weight"].to_numpy(), 1 - TOP)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as fh:
        pickle.dump({"model": m2, "feats": FEATS, "cutoff": cutoff, "train_end": TRAIN_END.isoformat()}, fh)
    report = {"trained": time.time(), "train_end": TRAIN_END.isoformat(), "pumps": int(M["label"].sum()),
              "coins": len(frames), "cutoff": cutoff, "test": test}
    REPORT_PATH.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


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


def reasons(r):
    out = []
    if r["pumps_prev_30d"] >= 1:
        out.append(f"pumped {int(r['pumps_prev_30d'])}× in 30d")
    if r["vol24_x"] >= 1.5:
        out.append(f"volume {r['vol24_x']:.1f}×")
    if r["from_30d_low"] >= 0.25:
        out.append(f"+{r['from_30d_low'] * 100:.0f}% off its 30d low")
    if 0 <= r["vs_ema200"] <= 0.10:
        out.append("just above EMA200")
    if r["range4h_x"] >= 1.5:
        out.append(f"candles {r['range4h_x']:.1f}× bigger")
    if r["age_days"] < 60:
        out.append("new coin")
    if r["ret_7d"] >= 0.10:
        out.append(f"+{r['ret_7d'] * 100:.0f}% this week")
    return out[:3]


def scan(symbols):
    m = model()
    if m is None:
        return None
    frames = _all_frames(sorted(set(symbols) | {"BTCUSDT"}))
    if "BTCUSDT" not in frames:
        return None
    btc, phase = frames["BTCUSDT"]["c"].ffill(), alt_phase(frames)
    meta = exchange.symbols()
    train_end = pd.Timestamp(m["train_end"])
    picks = []
    for s, df in frames.items():
        if s == "BTCUSDT" or s not in symbols:
            continue
        f = features(df, btc, phase, meta.get(s, {}).get("listed"))
        f = f[(f.index >= train_end) & (f["size_log10"] >= MIN_SIZE_LOG)].dropna(subset=["vol24_x"])
        if f.empty:
            continue
        f["score"] = m["model"].predict_proba(f[m["feats"]])[:, 1]
        c = df["c"].to_numpy()
        h = df["h"].fillna(df["c"]).to_numpy()
        l = df["l"].fillna(df["c"]).to_numpy()
        last_t = None
        for t, r in f[f["score"] >= m["cutoff"]].iterrows():
            if last_t is not None and t - last_t < pd.Timedelta(hours=24):
                continue
            last_t = t
            i = df.index.get_loc(t)
            e = c[i]
            nxt = slice(i + 1, min(len(c), i + 1 + WIN))
            done = i + WIN < len(c)
            close_t = t + H4
            picks.append({
                "id": f"{s}|{int(close_t.timestamp())}", "symbol": s,
                "time": int(close_t.timestamp()) - 900,         # same convention as alerts: candle open + 15m = close
                "price": float(e), "score": float(r["score"]), "x_chance": float(r["score"] / m["cutoff"]),
                "reasons": reasons(r),
                "peak": float(np.nanmax(h[nxt]) / e - 1) if i + 1 < len(c) else None,
                "trough": float(np.nanmin(l[nxt]) / e - 1) if i + 1 < len(c) else None,
                "now": float(df["c"].dropna().iloc[-1] / e - 1),
                "done": bool(done), "ret_24h": float(c[i + WIN] / e - 1 - FEE) if done else None,
            })
    fin = [p for p in picks if p["done"]]
    rec = {"n": len(fin)}
    if fin:
        pk = np.array([p["peak"] for p in fin])
        tr = np.array([p["trough"] for p in fin])
        rec.update(hit30=float((pk >= 0.30).mean()), hit15=float((pk >= 0.15).mean()), drop15=float((tr <= -0.15).mean()),
                   avg_24h=float(np.mean([p["ret_24h"] for p in fin])), since=int(train_end.timestamp()))
    return {"picks": picks, "record": rec, "built": time.time(),
            "report": json.loads(REPORT_PATH.read_text()) if REPORT_PATH.exists() else None}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if "--train" in sys.argv:
        train()
    else:
        print(__doc__)
