"""
Pre-pump signal research.

Computes indicators from data available *at* each 15m candle (no look-ahead),
measures which ones precede a 30%+ move within 24h, trains a gradient-boosted
model on the earlier part of the history and evaluates it on the later part
it never saw.

    python signals.py                # research report + saves models/signal_model.pkl
    python signals.py --test-days 21 --threshold 0.3

This is a statistical study, not trading advice: a "signal" means the odds of
a pump were historically higher than average, nothing more.
"""

import argparse
import json
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import scanner

MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "signal_model.pkl"
REPORT_PATH = MODEL_DIR / "signal_report.json"

H1, H4, H24, D7 = 4, 16, 96, 672  # in 15m candles
WARMUP = D7                        # need 7 days of history before features are valid

FEATURES = {
    "ret_1h": "Price change, last 1h",
    "ret_4h": "Price change, last 4h",
    "ret_24h": "Price change, last 24h",
    "ret_7d": "Price change, last 7d",
    "vol_ratio_1h": "Volume last 1h vs 7d hourly average",
    "vol_ratio_4h": "Volume last 4h vs 7d average",
    "vol_ratio_24h": "Volume last 24h vs 7d daily average",
    "volatility_24h": "Volatility of 15m returns, last 24h",
    "vol_compression": "24h volatility / 7d volatility",
    "range_pos_7d": "Position inside 7d high-low range (0=low, 1=high)",
    "from_7d_high": "Distance below 7d high",
    "rsi_15m": "RSI(14) on 15m candles",
    "rsi_1h": "RSI(14) on 1h (approx.)",
    "liquidity_24h": "log10 of 24h USDT volume",
    "past_pumps_30d": "Share of the last 30d already labeled as pump candles",
    "btc_ret_4h": "BTC price change, last 4h",
    "btc_ret_24h": "BTC price change, last 24h",
    "hour": "Hour of day (UTC)",
}
FEATURE_COLS = list(FEATURES)


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def load_full(path):
    """Like scanner.load_candles but with the trade/taker columns signals need."""
    cols = ["open_time", "open", "high", "low", "close", "quote_volume", "trades", "taker_buy_quote"]
    df = pd.read_csv(path, usecols=cols)
    if len(df) < 2:
        return None
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.set_index("open_time")
    df = df.reindex(pd.date_range(df.index[0], df.index[-1], freq="15min"))
    df.index.name = "open_time"
    return df


def rsi(close, n):
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / down.replace(0, np.nan))


def compute_features(df, btc_close=None, threshold=0.30):
    """
    All features at row t use candles <= t only. `past_pumps_30d` uses labels
    shifted by 24h, since a label at t'' is only known once t''+24h has passed.
    """
    c, qv = df["close"].ffill(), df["quote_volume"].fillna(0)
    f = pd.DataFrame(index=df.index)

    f["ret_1h"] = c / c.shift(H1) - 1
    f["ret_4h"] = c / c.shift(H4) - 1
    f["ret_24h"] = c / c.shift(H24) - 1
    f["ret_7d"] = c / c.shift(D7) - 1

    avg15 = qv.rolling(D7, min_periods=D7 // 2).mean()  # avg 15m volume over 7d
    f["vol_ratio_1h"] = qv.rolling(H1).sum() / (avg15 * H1)
    f["vol_ratio_4h"] = qv.rolling(H4).sum() / (avg15 * H4)
    f["vol_ratio_24h"] = qv.rolling(H24).sum() / (avg15 * H24)



    lr = np.log(c).diff()
    v24 = lr.rolling(H24, min_periods=H24 // 2).std()
    f["volatility_24h"] = v24
    f["vol_compression"] = v24 / lr.rolling(D7, min_periods=D7 // 2).std()

    hi7 = df["high"].rolling(D7, min_periods=D7 // 2).max()
    lo7 = df["low"].rolling(D7, min_periods=D7 // 2).min()
    f["range_pos_7d"] = (c - lo7) / (hi7 - lo7)
    f["from_7d_high"] = c / hi7 - 1

    f["rsi_15m"] = rsi(c, 14)
    f["rsi_1h"] = rsi(c, 56)
    f["liquidity_24h"] = np.log10(qv.rolling(H24).sum() + 1)

    fwd_max = df["high"][::-1].rolling(H24, min_periods=1).max()[::-1].shift(-1)
    label = (fwd_max / df["close"] - 1 >= threshold).astype(float)
    f["past_pumps_30d"] = label.shift(H24).rolling(30 * 96, min_periods=D7).mean()

    if btc_close is not None:
        b = btc_close.reindex(df.index).ffill()
        f["btc_ret_4h"] = b / b.shift(H4) - 1
        f["btc_ret_24h"] = b / b.shift(H24) - 1
    else:
        f["btc_ret_4h"] = f["btc_ret_24h"] = np.nan
    f["hour"] = df.index.hour

    # Targets (future — used for training/evaluation only, never as inputs)
    f["y"] = label
    f["fwd_max_gain"] = fwd_max / df["close"] - 1
    f["fwd_ret_24h"] = df["close"].shift(-H24) / df["close"] - 1
    f["fwd_min"] = df["low"][::-1].rolling(H24, min_periods=1).min()[::-1].shift(-1) / df["close"] - 1
    f["close"] = df["close"]

    f = f.replace([np.inf, -np.inf], np.nan)
    f.loc[df["close"].isna()] = np.nan  # no prediction on missing candles
    return f


def build_dataset(threshold, workers=8):
    paths = scanner.candle_paths()
    btc = load_full(scanner.CANDLE_DIR / "BTCUSDT.csv")
    btc_close = btc["close"] if btc is not None else None

    def one(path):
        df = load_full(path)
        if df is None or len(df) < WARMUP + H24:
            return None
        f = compute_features(df, btc_close, threshold).iloc[WARMUP:]
        f = f[f["close"].notna()]
        f["symbol"] = path.stem
        return f.astype({c: "float32" for c in FEATURE_COLS + ["fwd_max_gain", "fwd_ret_24h", "fwd_min"]})

    with ThreadPoolExecutor(max_workers=workers) as pool:
        parts = [p for p in pool.map(one, paths) if p is not None]
    data = pd.concat(parts)
    data.index.name = "time"
    return data.reset_index()


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #
def distinct_signals(df, score_col, cutoff, cooldown_h=24):
    """One signal per coin per cooldown window: the first candle whose score >= cutoff."""
    hits = df[df[score_col] >= cutoff].sort_values(["symbol", "time"])
    keep, last = [], {}
    cool = pd.Timedelta(hours=cooldown_h)
    for idx, sym, t in zip(hits.index, hits["symbol"], hits["time"]):
        if sym not in last or t - last[sym] >= cool:
            keep.append(idx)
            last[sym] = t
    return df.loc[keep]


def summarize_signals(sig):
    if sig.empty:
        return {"signals": 0}
    r = sig["fwd_ret_24h"].dropna()
    return {
        "signals": int(len(sig)),
        "hit_rate": float(sig["y"].mean()),
        "median_max_gain": float(sig["fwd_max_gain"].median()),
        "median_ret_24h": float(r.median()) if len(r) else None,
        "mean_ret_24h": float(r.mean()) if len(r) else None,
        "share_negative_24h": float((r < 0).mean()) if len(r) else None,
        "median_max_drawdown": float(sig["fwd_min"].median()),
    }


FEE = 0.001  # taker fee per side (kept from the original study for comparability)


def simulate(sig, tp_list=(0.10, 0.20, 0.30), sl_list=(-0.05, -0.10, None)):
    """
    Enter at the signal candle's close, then walk the next 24h of candles:
    stop-loss if the low touches it (checked first — pessimistic when both
    hit in one candle), take-profit if the high touches it, else exit at the
    close 24h later. Returns mean/median net return and win rate per combo.
    """
    paths = {}
    for sym in sig["symbol"].unique():
        df = load_full(scanner.CANDLE_DIR / f"{sym}.csv")
        paths[sym] = df[["high", "low", "close"]]

    results = []
    for tp in tp_list:
        for sl in sl_list:
            rets = []
            for sym, t, entry in zip(sig["symbol"], sig["time"], sig["close"]):
                win = paths[sym].loc[t + pd.Timedelta(minutes=15): t + pd.Timedelta(hours=24)].dropna()
                if win.empty:
                    continue
                r = win["close"].iloc[-1] / entry - 1
                for lo, hi in zip(win["low"].to_numpy(), win["high"].to_numpy()):
                    if sl is not None and lo / entry - 1 <= sl:
                        r = sl
                        break
                    if hi / entry - 1 >= tp:
                        r = tp
                        break
                rets.append(r - 2 * FEE)
            rets = np.array(rets)
            results.append({
                "tp": tp, "sl": sl, "trades": int(len(rets)),
                "mean": float(rets.mean()) if len(rets) else None,
                "median": float(np.median(rets)) if len(rets) else None,
                "win_rate": float((rets > 0).mean()) if len(rets) else None,
                "total": float(rets.sum()) if len(rets) else None,
            })
    return results


def lift_table(train, col, bins=10):
    """Pump rate per decile of a feature (training data only)."""
    s = train[[col, "y"]].dropna()
    if s[col].nunique() < bins:
        return None
    q = pd.qcut(s[col], bins, duplicates="drop")
    t = s.groupby(q, observed=True)["y"].agg(["mean", "size"])
    return [{"lo": float(iv.left), "hi": float(iv.right), "rate": float(m), "n": int(n)}
            for iv, (m, n) in zip(t.index, t.values)]


# --------------------------------------------------------------------------- #
def run(threshold, test_days, seed=7):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    t0 = time.time()
    print("Building features for all pairs…")
    data = build_dataset(threshold)
    data = data[data["fwd_ret_24h"].notna() | data["y"].notna()]
    labeled = data[data["time"] <= data["time"].max() - pd.Timedelta(hours=24)]  # full 24h future known
    print(f"  {len(labeled):,} rows, {labeled['symbol'].nunique()} pairs ({time.time() - t0:.0f}s)")

    split = labeled["time"].max() - pd.Timedelta(days=test_days)
    # 24h embargo so no training label overlaps the test period
    train = labeled[labeled["time"] < split - pd.Timedelta(hours=24)]
    test = labeled[labeled["time"] >= split]
    base_train, base_test = train["y"].mean(), test["y"].mean()
    print(f"  train {train['time'].min():%b %d} – {train['time'].max():%b %d}: "
          f"{len(train):,} rows, pump rate {base_train:.2%}")
    print(f"  test  {test['time'].min():%b %d} – {test['time'].max():%b %d}: "
          f"{len(test):,} rows, pump rate {base_test:.2%}")

    # Single-indicator lift (training data)
    indicators = []
    for col in FEATURE_COLS:
        tbl = lift_table(train, col)
        if not tbl:
            continue
        top, bottom = tbl[-1], tbl[0]
        best = max(tbl, key=lambda r: r["rate"])
        indicators.append({
            "feature": col, "description": FEATURES[col], "deciles": tbl,
            "best_decile_rate": best["rate"], "best_decile_lift": best["rate"] / base_train,
            "best_decile": [best["lo"], best["hi"]],
            "top_decile_lift": top["rate"] / base_train, "bottom_decile_lift": bottom["rate"] / base_train,
        })
    indicators.sort(key=lambda r: -r["best_decile_lift"])

    # Model: all positives + a sample of negatives (reweighted) to keep training fast
    rng = np.random.default_rng(seed)
    neg = train[train["y"] == 0]
    neg = neg.iloc[rng.choice(len(neg), size=min(len(neg), 600_000), replace=False)]
    tr = pd.concat([train[train["y"] == 1], neg])
    w = np.where(tr["y"] == 1, 1.0, len(train[train["y"] == 0]) / len(neg))
    model = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
        l2_regularization=1.0, random_state=seed)
    print("Training model…")
    model.fit(tr[FEATURE_COLS], tr["y"], sample_weight=w)

    test = test.copy()
    test["score"] = model.predict_proba(test[FEATURE_COLS])[:, 1]
    auc = roc_auc_score(test["y"], test["score"])

    # Precision by score percentile (test)
    tiers, tier_signals = [], {}
    for top_pct in (0.1, 0.5, 1, 2, 5):
        cut = test["score"].quantile(1 - top_pct / 100)
        sig = distinct_signals(test, "score", cut)
        tier_signals[top_pct] = sig
        tiers.append({"top_pct": top_pct, "cutoff": float(cut),
                      "candle_precision": float(test.loc[test["score"] >= cut, "y"].mean()),
                      **summarize_signals(sig)})

    # Baseline to compare against: random coin/time with the same cooldown logic
    rand = test.sample(n=min(len(test), 3000), random_state=seed)
    baseline = summarize_signals(rand)

    print("Simulating take-profit / stop-loss exits…")
    trading = {
        "top 0.5%": simulate(tier_signals[0.5]),
        "top 1%": simulate(tier_signals[1]),
        "random": simulate(rand.sample(n=1000, random_state=seed)),
    }

    # Simple, transparent rule using the strongest single indicators
    rule_mask = (test["vol_ratio_4h"] >= 3) & (test["ret_4h"] > 0.03) & (test["liquidity_24h"] < 7)
    rule = summarize_signals(distinct_signals(test.assign(r=rule_mask.astype(float)), "r", 1.0))

    # Final model trained on everything (for live scoring); tiers keep their test-set cutoffs
    final_neg = labeled[labeled["y"] == 0]
    final_neg = final_neg.iloc[rng.choice(len(final_neg), size=min(len(final_neg), 800_000), replace=False)]
    full = pd.concat([labeled[labeled["y"] == 1], final_neg])
    wf = np.where(full["y"] == 1, 1.0, len(labeled[labeled["y"] == 0]) / len(final_neg))
    final = HistGradientBoostingClassifier(**model.get_params()).fit(full[FEATURE_COLS], full["y"], sample_weight=wf)
    # Recompute tier cutoffs for the final model on the same test window (in-sample now, used only to bucket)
    test_scores_final = final.predict_proba(test[FEATURE_COLS])[:, 1]
    for t in tiers:
        t["final_cutoff"] = float(np.quantile(test_scores_final, 1 - t["top_pct"] / 100))

    report = {
        "created": int(time.time()),
        "threshold": threshold, "window_hours": 24,
        "train": {"start": str(train["time"].min()), "end": str(train["time"].max()),
                  "rows": int(len(train)), "pump_rate": float(base_train)},
        "test": {"start": str(test["time"].min()), "end": str(test["time"].max()),
                 "rows": int(len(test)), "pump_rate": float(base_test), "auc": float(auc)},
        "tiers": tiers, "random_baseline": baseline, "simple_rule": rule, "trading": trading,
        "indicators": indicators,
    }
    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as fh:
        pickle.dump({"model": final, "features": FEATURE_COLS, "threshold": threshold, "tiers": tiers}, fh)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print_report(report)
    print(f"\nSaved {MODEL_PATH} and {REPORT_PATH} ({time.time() - t0:.0f}s total)")
    return report


def print_report(r):
    line = "=" * 86
    pct = lambda v: "—" if v is None else f"{v * 100:+.1f}%"
    print(f"\n{line}\nSINGLE INDICATORS — pump rate in the best decile vs. average (training data)\n{line}")
    base = r["train"]["pump_rate"]
    print(f"Average pump rate: {base:.2%} of candles\n")
    for ind in r["indicators"]:
        lo, hi = ind["best_decile"]
        print(f"  {ind['description'][:46]:<46} best decile [{lo:9.3g}, {hi:9.3g}]  "
              f"{ind['best_decile_rate']:6.2%}  ({ind['best_decile_lift']:4.1f}x)")

    t = r["test"]
    print(f"\n{line}\nMODEL — out-of-sample test {t['start'][:10]} → {t['end'][:10]}  "
          f"(AUC {t['auc']:.3f}, base rate {t['pump_rate']:.2%})\n{line}")
    print("Signals = first time a coin's score enters the tier, max one per coin per 24h.\n")
    print(f"  {'tier':<10}{'signals':>8}{'hit rate':>10}{'lift':>7}{'med max gain':>14}"
          f"{'med 24h ret':>13}{'mean 24h ret':>14}{'% red 24h':>11}{'med drawdown':>14}")
    rows = [(f"top {x['top_pct']}%", x) for x in r["tiers"]] + [
        ("rule", r["simple_rule"]), ("random", r["random_baseline"])]
    for name, x in rows:
        if not x.get("signals"):
            print(f"  {name:<10}{0:>8}")
            continue
        print(f"  {name:<10}{x['signals']:>8}{x['hit_rate']:>10.1%}{x['hit_rate'] / t['pump_rate']:>6.1f}x"
              f"{pct(x['median_max_gain']):>14}{pct(x['median_ret_24h']):>13}{pct(x['mean_ret_24h']):>14}"
              f"{x['share_negative_24h']:>11.0%}{pct(x['median_max_drawdown']):>14}")
    print("\n  rule = 4h volume ≥ 3× normal AND price up >3% in 4h AND 24h volume < $10M")
    print("  'hit' = price touched +30% within 24h. '24h ret' = close 24h later (what holding would give).")

    print(f"\n{line}\nTRADING SIMULATION — enter at signal close, exit at TP / SL / 24h, 0.1% fee per side\n{line}")
    print(f"  {'signals':<10}{'TP':>6}{'SL':>7}{'trades':>8}{'win rate':>10}{'mean/trade':>12}{'median':>9}{'sum':>9}")
    for name, rows in r["trading"].items():
        for x in rows:
            sl = "none" if x["sl"] is None else f"{x['sl'] * 100:.0f}%"
            print(f"  {name:<10}{x['tp'] * 100:>5.0f}%{sl:>7}{x['trades']:>8}{x['win_rate']:>10.0%}"
                  f"{pct(x['mean']):>12}{pct(x['median']):>9}{x['total'] * 100:>+8.0f}%")
        print()


def live_scores(frames=None, top=None):
    """
    Score the latest closed candle of every pair with the saved model.
    `frames` maps symbol -> load_full() frame; read from disk when omitted.
    Returns rows sorted by score, each tagged with the backtest tier it falls in.
    """
    with open(MODEL_PATH, "rb") as fh:
        saved = pickle.load(fh)
    model, tiers = saved["model"], saved["tiers"]
    if frames is None:
        frames = {p.stem: load_full(p) for p in scanner.candle_paths()}
        frames = {s: f for s, f in frames.items() if f is not None}
    btc = frames.get("BTCUSDT")
    btc_close = btc["close"] if btc is not None else None

    def latest(item):
        sym, df = item
        if len(df) < WARMUP + 1:
            return None
        # Only the tail is needed: 30d for past_pumps_30d plus a little slack
        f = compute_features(df.iloc[-(31 * 96):], btc_close, saved["threshold"])
        if np.isnan(f["close"].iloc[-1]):
            return None
        return sym, df.index[-1], f[FEATURE_COLS + ["close"]].iloc[-1]

    with ThreadPoolExecutor(max_workers=8) as pool:
        last = [x for x in pool.map(latest, frames.items()) if x is not None]
    if not last:
        return []
    X = pd.DataFrame([x[2] for x in last])
    scores = model.predict_proba(X[FEATURE_COLS])[:, 1]  # one batched call

    rows = []
    for (sym, t, feats), score in zip(last, scores):
        score = float(score)
        tier = next((t_ for t_ in tiers if score >= t_["final_cutoff"]), None)
        rows.append({"symbol": sym, "time": t, "score": score,
                     "tier": tier["top_pct"] if tier else None,
                     "hist_hit_rate": tier["hit_rate"] if tier else None,
                     **{k: float(feats[k]) for k in FEATURE_COLS}, "price": float(feats["close"])})
    rows.sort(key=lambda r: -r["score"])
    return rows[:top] if top else rows


def print_live(rows, n=15):
    line = "=" * 86
    print(f"{line}\nVOLATILITY WATCHLIST — latest candle {rows[0]['time']:%Y-%m-%d %H:%M} UTC\n{line}")
    print("Higher score = historically higher odds of a +30% touch in 24h. NOT a buy signal:")
    print("in backtests these coins were as likely to fall as rise, and had no edge after fees.\n")
    print(f"  {'coin':<14}{'score':>7}{'tier':>9}{'hist hit':>10}{'4h chg':>9}{'24h chg':>9}{'vol 4h':>8}{'24h vol':>10}")
    for r in rows[:n]:
        tier = f"top {r['tier']}%" if r["tier"] else "—"
        hit = f"{r['hist_hit_rate']:.0%}" if r["hist_hit_rate"] else "—"
        print(f"  {r['symbol']:<14}{r['score']:>7.3f}{tier:>9}{hit:>10}{r['ret_4h'] * 100:>+8.1f}%"
              f"{r['ret_24h'] * 100:>+8.1f}%{r['vol_ratio_4h']:>7.1f}x{10 ** r['liquidity_24h'] / 1e6:>9.1f}M")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--threshold", type=float, default=0.30)
    p.add_argument("--test-days", type=int, default=28)
    p.add_argument("--live", action="store_true", help="score the latest candles with the saved model")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.live:
        if not MODEL_PATH.exists():
            sys.exit("No saved model yet — run `python signals.py` first.")
        print_live(live_scores())
    else:
        run(args.threshold, args.test_days)


if __name__ == "__main__":
    main()
