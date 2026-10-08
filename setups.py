"""
Rule-based 15m setups (volume + MACD + EMA) and their historical outcomes.

Two alert types:
  ignition  - the first explosive candle: volume spike, strong green candle,
              MACD histogram positive, price above EMA50, not already pumped.
  trend     - the confirmed move (e.g. GTC 2026-10-04 18:30): EMAs stacked
              9 > 21 > 50 > 200, MACD above signal and zero, volume elevated on
              this candle and over the last 2h, green candle closing at a new high.

    python setups.py               # backtest both setups over all downloaded data
    python setups.py --symbol GTCUSDT --since 2026-10-04

Everything at candle t uses only candles <= t. An alert fires at the CLOSE of
the candle, which is the earliest moment you could act on it.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import scanner
import signals

REPORT_PATH = Path(__file__).parent / "models" / "setups_report.json"
FEE = 0.001
COOLDOWN_H = 6  # one alert per coin per setup per 6h

DEFAULTS = {
    "ignition": {"vol_x": 5.0, "min_chg": 0.03, "max_ret_24h": 0.15},
    "trend": {"vol_x": 3.0, "vol_x_2h": 2.0, "min_chg": 0.01, "high_lookback": 8},
}
SETUP_NAMES = {"ignition": "Ignition", "trend": "Trend confirmation",
               "trend_confirmed": "Trend, confirmed 1h later", "ignition_confirmed": "Ignition, confirmed 1h later"}

# 1-hour follow-through check: an alert is "confirmed" if over the next 4 candles
# price never dipped more than 1% below the alert close, made a higher high than
# the alert candle, and the 4th candle closed above EMA9.
CONFIRM_CANDLES = 4
CONFIRM_MAX_DIP = 0.01


# --------------------------------------------------------------------------- #
def indicators(df):
    """EMA 9/21/50/200, MACD(12,26,9), volume multiple vs prior 24h, candle change."""
    c = df["close"].ffill()
    out = pd.DataFrame(index=df.index)
    for n in (9, 21, 50, 200):
        out[f"ema{n}"] = c.ewm(span=n, adjust=False, min_periods=n).mean()
    fast, slow = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    out["macd"] = fast - slow
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]
    qv = df["quote_volume"].fillna(0)
    avg = qv.rolling(96, min_periods=48).mean().shift(1)  # previous 24h, excluding this candle
    out["vol_x"] = qv / avg
    out["vol_x_2h"] = qv.rolling(8).mean() / avg
    out["chg"] = df["close"] / df["open"] - 1
    out["ret_24h"] = c / c.shift(96) - 1
    out["prev_high_close"] = c.shift(1).rolling(8).max()
    out["rsi"] = signals.rsi(c, 14)
    out["close"] = df["close"]
    return out


def detect(ind, params=None):
    """Boolean Series per setup for every candle."""
    p = {k: {**v, **((params or {}).get(k, {}))} for k, v in DEFAULTS.items()}
    i, t = p["ignition"], p["trend"]
    green = ind["chg"] > 0
    ignition = (
        (ind["vol_x"] >= i["vol_x"]) & (ind["chg"] >= i["min_chg"])
        & (ind["macd_hist"] > 0) & (ind["close"] > ind["ema50"])
        & (ind["ret_24h"] < i["max_ret_24h"])
    )
    stacked = (ind["ema9"] > ind["ema21"]) & (ind["ema21"] > ind["ema50"]) & (ind["ema50"] > ind["ema200"])
    trend = (
        stacked & (ind["macd"] > ind["macd_signal"]) & (ind["macd"] > 0)
        & (ind["vol_x"] >= t["vol_x"]) & (ind["vol_x_2h"] >= t["vol_x_2h"])
        & green & (ind["chg"] >= t["min_chg"])
        & (ind["close"] >= ind["prev_high_close"])
    )
    return {"ignition": ignition.fillna(False), "trend": trend.fillna(False)}


def apply_cooldown(mask, hours=COOLDOWN_H):
    times = mask.index[mask.to_numpy()]
    keep, last = [], None
    for t in times:
        if last is None or t - last >= pd.Timedelta(hours=hours):
            keep.append(t)
            last = t
    return keep


def outcomes(df, t):
    """What happened in the 24h after entering at the close of candle t."""
    entry = df.at[t, "close"]
    w = df.loc[t + pd.Timedelta(minutes=15): t + pd.Timedelta(hours=24)].dropna(subset=["close"])
    if len(w) < 90:  # need (almost) the full 24h to judge
        return None
    hi, lo = w["high"].to_numpy() / entry - 1, w["low"].to_numpy() / entry - 1
    first = lambda arr, cond: int(np.argmax(cond)) if cond.any() else None
    i5 = first(hi, hi >= 0.05)
    return {
        "max_gain": float(hi.max()), "max_drawdown": float(lo.min()),
        "ret_4h": float(w["close"].iloc[min(15, len(w) - 1)] / entry - 1),
        "ret_24h": float(w["close"].iloc[-1] / entry - 1),
        "hit5": bool(hi.max() >= 0.05), "hit10": bool(hi.max() >= 0.10), "hit20": bool(hi.max() >= 0.20),
        # did price fall 5% before it rose 5%? (the trade most people would get stopped out of)
        "dd_before_5": float(lo[: i5 + 1].min()) if i5 is not None else float(lo.min()),
        "hi": hi, "lo": lo, "close_path": w["close"].to_numpy() / entry - 1,
    }


def confirmation(df, t):
    """1h follow-through of an alert at candle t; None while the hour isn't over."""
    i = df.index.get_loc(t)
    nxt = df.iloc[i + 1: i + 1 + CONFIRM_CANDLES]
    if len(nxt) < CONFIRM_CANDLES or nxt["close"].isna().any():
        return None
    p0 = df["close"].iat[i]
    c1 = nxt["close"].iat[-1]
    ema9 = df["close"].iloc[max(0, i - 200): i + 1 + CONFIRM_CANDLES].ffill().ewm(span=9, adjust=False).mean().iat[-1]
    held = bool(nxt["low"].min() >= p0 * (1 - CONFIRM_MAX_DIP))
    higher_high = bool(nxt["high"].max() > df["high"].iat[i])
    above_ema9 = bool(c1 > ema9)
    return {"chg_1h": float(c1 / p0 - 1), "price_1h": float(c1), "time_1h": nxt.index[-1],
            "held": held, "higher_high": higher_high, "above_ema9": above_ema9,
            "confirmed": held and higher_high and above_ema9}


def simulate(rows, tp, sl):
    rets = []
    for o in rows:
        r = o["close_path"][-1]
        for h, l in zip(o["hi"], o["lo"]):
            if sl is not None and l <= sl:
                r = sl
                break
            if h >= tp:
                r = tp
                break
        rets.append(r - 2 * FEE)
    rets = np.array(rets)
    return {"tp": tp, "sl": sl, "trades": int(len(rets)), "mean": float(rets.mean()),
            "median": float(np.median(rets)), "win_rate": float((rets > 0).mean())}


def summarize(rows):
    if not rows:
        return {"alerts": 0}
    a = lambda k: np.array([o[k] for o in rows])
    return {
        "alerts": len(rows),
        "hit5": float(a("hit5").mean()), "hit10": float(a("hit10").mean()), "hit20": float(a("hit20").mean()),
        "median_max_gain": float(np.median(a("max_gain"))),
        "median_ret_4h": float(np.median(a("ret_4h"))), "median_ret_24h": float(np.median(a("ret_24h"))),
        "mean_ret_24h": float(a("ret_24h").mean()), "share_down_24h": float((a("ret_24h") < 0).mean()),
        "median_drawdown": float(np.median(a("max_drawdown"))),
        "share_dd5_first": float((a("dd_before_5") <= -0.05).mean()),
    }


# --------------------------------------------------------------------------- #
def backtest(params=None, workers=8, paths=None, report_path=None):
    """`paths` defaults to every tracked coin's candle CSV."""
    paths = scanner.candle_paths() if paths is None else paths
    report_path = report_path or REPORT_PATH
    rng = np.random.default_rng(7)

    def one(path):
        df = signals.load_full(path)
        if df is None or len(df) < 400:
            return None
        ind = indicators(df)
        masks = detect(ind, params)
        res = {}
        for name, m in masks.items():
            m = m.copy()
            m.iloc[:300] = False  # EMA200 warm-up
            res[name], res[f"{name}_confirmed"] = [], []
            for t in apply_cooldown(m):
                o = outcomes(df, t)
                if o:
                    res[name].append({"symbol": path.stem, "time": t, **o})
                # Confirmed variant: only alerts that held for 1h, entered 1h later
                c = confirmation(df, t)
                if c and c["confirmed"]:
                    o1 = outcomes(df, c["time_1h"])
                    if o1:
                        res[f"{name}_confirmed"].append({"symbol": path.stem, "time": c["time_1h"], **o1})
        # Random baseline: a few random candles per coin
        valid = df.index[300:-100]
        res["random"] = []
        for t in rng.choice(valid, size=min(12, len(valid)), replace=False):
            o = outcomes(df, pd.Timestamp(t))
            if o:
                res["random"].append({"symbol": path.stem, "time": pd.Timestamp(t), **o})
        return res

    with ThreadPoolExecutor(max_workers=workers) as pool:
        parts = [r for r in pool.map(one, paths) if r]
    allrows = {k: [o for p in parts for o in p[k]]
               for k in ("ignition", "trend", "ignition_confirmed", "trend_confirmed", "random")}

    report = {"created": int(time.time()), "params": {k: {**v, **((params or {}).get(k, {}))} for k, v in DEFAULTS.items()},
              "cooldown_h": COOLDOWN_H, "setups": {}}
    for name, rows in allrows.items():
        rows.sort(key=lambda o: o["time"])
        mid = len(rows) // 2
        sims = [simulate(rows, tp, sl) for tp, sl in ((0.05, -0.03), (0.05, -0.05), (0.10, -0.05), (0.10, None), (0.20, -0.10))]
        report["setups"][name] = {
            "all": summarize(rows),
            # Stability check: does it hold in both halves of the period?
            "first_half": summarize(rows[:mid]), "second_half": summarize(rows[mid:]),
            "sims": sims,
            "per_day": float(len(rows) / 90) if name != "random" else None,
            "examples": [{"symbol": o["symbol"], "time": int(o["time"].timestamp()), "max_gain": o["max_gain"],
                          "ret_24h": o["ret_24h"], "max_drawdown": o["max_drawdown"]}
                         for o in sorted(rows, key=lambda o: -o["time"].timestamp())[:300]],
        }
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    return report


def print_report(r):
    line = "=" * 92
    pct = lambda v: f"{v * 100:+.1f}%"
    print(f"{line}\nSETUP BACKTEST — alert at candle close, outcome over the next 24h, all pairs, ~90 days\n{line}")
    print(f"{'':<30}{'alerts':>7}{'/day':>6}{'≥+5%':>7}{'≥+10%':>7}{'≥+20%':>7}{'med 4h':>8}{'med 24h':>9}"
          f"{'% down':>8}{'med DD':>8}{'-5% first':>10}")
    for name, s in r["setups"].items():
        for part in ("all", "first_half", "second_half"):
            x = s[part]
            label = (SETUP_NAMES.get(name, "Random entry") if part == "all" else f"  {part.replace('_', ' ')}")
            if not x.get("alerts"):
                print(f"{label:<30}{0:>7}")
                continue
            per_day = f"{s['per_day']:.1f}" if (part == "all" and s["per_day"]) else ""
            print(f"{label:<30}{x['alerts']:>7}{per_day:>6}{x['hit5']:>7.0%}{x['hit10']:>7.0%}{x['hit20']:>7.0%}"
                  f"{pct(x['median_ret_4h']):>8}{pct(x['median_ret_24h']):>9}{x['share_down_24h']:>8.0%}"
                  f"{pct(x['median_drawdown']):>8}{x['share_dd5_first']:>10.0%}")
        print()
    print("≥+5% = price touched +5% within 24h.  -5% first = fell 5% before ever touching +5%.")
    print(f"\n{line}\nTRADE SIMULATION — take-profit / stop-loss / exit after 24h, 0.1% fee per side\n{line}")
    for name, s in r["setups"].items():
        print(f"{SETUP_NAMES.get(name, 'Random entry')}:")
        for x in s["sims"]:
            sl = "none" if x["sl"] is None else f"{x['sl'] * 100:.0f}%"
            print(f"   TP +{x['tp'] * 100:.0f}%  SL {sl:>5}   {x['trades']:>5} trades   win {x['win_rate']:.0%}   "
                  f"mean {pct(x['mean'])}   median {pct(x['median'])}")


def show_symbol(symbol, since):
    df = signals.load_full(scanner.CANDLE_DIR / f"{symbol}.csv")
    ind = indicators(df)
    masks = detect(ind)
    for name, m in masks.items():
        times = [t for t in apply_cooldown(m) if t >= pd.Timestamp(since, tz="UTC")]
        print(f"{SETUP_NAMES[name]}: " + (", ".join(f"{t:%m-%d %H:%M}" for t in times) or "none"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbol")
    p.add_argument("--since", default="2026-01-01")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.symbol:
        show_symbol(args.symbol, args.since)
    else:
        t0 = time.time()
        r = backtest()
        print_report(r)
        print(f"\nSaved {REPORT_PATH} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
