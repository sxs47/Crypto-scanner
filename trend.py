"""
Daily time-series momentum (trend following) on liquid Binance USDT coins.

Each day, among the N most-traded coins (by trailing 30-day volume, using only
coins that existed then), hold those in an uptrend — EMA(fast) > EMA(slow) and
close > EMA(slow) — sized by inverse volatility, total exposure capped at 100%.
Otherwise cash. Signals use the day's close; positions start the next day.

    python trend.py --download     # fetch/refresh daily candles (all history)
    python trend.py                # backtest + current signals

Caveat: only coins listed today are in the data (delisted coins are missing),
which flatters every long-only crypto backtest, including the benchmarks.
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

DAILY_DIR = scanner.DATA_DIR / "daily"
REPORT_PATH = Path(__file__).parent / "models" / "trend_report.json"
FEE = 0.001          # per side, on traded value
DAY_MS = 86_400_000
DEFAULT = {"fast": 20, "slow": 50, "top_n": 30, "vol_days": 20}
GRID = [{"fast": 10, "slow": 30}, {"fast": 20, "slow": 50}, {"fast": 50, "slow": 100}]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def download_symbol(symbol):
    path = DAILY_DIR / f"{symbol}.csv"
    existing = pd.read_csv(path) if path.exists() else None
    start = 0
    if existing is not None and len(existing):
        start = int(pd.Timestamp(existing["date"].iloc[-1], tz="UTC").timestamp() * 1000) + DAY_MS
    now = int(time.time() * 1000)
    end = now - now % DAY_MS  # only closed days
    rows = []
    while start < end:
        batch = scanner.api_get("/api/v3/klines", {"symbol": symbol, "interval": "1d",
                                                   "startTime": start, "endTime": end - 1, "limit": 1000})
        if not batch:
            break
        rows.extend(batch)
        start = batch[-1][0] + DAY_MS
        if len(batch) < 1000:
            break
    if rows:
        new = pd.DataFrame({
            "date": pd.to_datetime([r[0] for r in rows], unit="ms").strftime("%Y-%m-%d"),
            "open": [float(r[1]) for r in rows], "high": [float(r[2]) for r in rows],
            "low": [float(r[3]) for r in rows], "close": [float(r[4]) for r in rows],
            "quote_volume": [float(r[7]) for r in rows],
        })
        df = pd.concat([existing, new]) if existing is not None else new
        df.drop_duplicates("date", keep="last").to_csv(path, index=False)
    return symbol, len(rows)


def download_all(workers=8, progress=None):
    if scanner._base_url is None:
        scanner._base_url = scanner.pick_base_url()
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    symbols = scanner.get_usdt_symbols()
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for sym, n in pool.map(download_symbol, symbols):
            done += 1
            if progress:
                progress(done, len(symbols), sym)
    return len(symbols)


def load_panel():
    """Wide frames (date x symbol) of close and quote volume."""
    skip = scanner.excluded_symbols()
    closes, vols = {}, {}
    for p in sorted(DAILY_DIR.glob("*.csv")):
        if p.stem in skip or p.stem.removesuffix("USDT") in scanner.EXCLUDED_BASES:
            continue
        d = pd.read_csv(p, parse_dates=["date"]).set_index("date")
        if len(d) < 60:
            continue
        closes[p.stem], vols[p.stem] = d["close"], d["quote_volume"]
    close = pd.DataFrame(closes).sort_index()
    vol = pd.DataFrame(vols).reindex(close.index)
    return close, vol


# --------------------------------------------------------------------------- #
# Strategy
# --------------------------------------------------------------------------- #
def signals_and_weights(close, vol, fast=20, slow=50, top_n=30, vol_days=20):
    ema_f = close.ewm(span=fast, adjust=False, min_periods=slow).mean()
    ema_s = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    uptrend = (ema_f > ema_s) & (close > ema_s)
    # Universe: top-N by trailing 30d dollar volume, among coins with a full slow-EMA history
    adv = vol.rolling(30, min_periods=20).mean().where(ema_s.notna())
    rank = adv.rank(axis=1, ascending=False)
    universe = rank <= top_n
    rets = close.pct_change(fill_method=None)
    sigma = rets.rolling(vol_days, min_periods=vol_days // 2).std()
    raw = (uptrend & universe).astype(float) / sigma
    raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0)
    # Inverse-vol weights over the selected coins; exposure scaled to how many of the
    # universe are in trend (all in trend -> 100% invested, none -> cash)
    share_in_trend = (uptrend & universe).sum(axis=1) / universe.sum(axis=1).replace(0, np.nan)
    w = raw.div(raw.sum(axis=1).replace(0, np.nan), axis=0).mul(share_in_trend.fillna(0), axis=0).fillna(0)
    return w, uptrend, universe, ema_f, ema_s, sigma


def backtest(close, vol, **params):
    w, *_ = signals_and_weights(close, vol, **params)
    rets = close.pct_change(fill_method=None).fillna(0)
    held = w.shift(1).fillna(0)                      # decide at close, hold from next day
    gross = (held * rets).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1).fillna(held.abs().sum(axis=1))
    net = gross - turnover * FEE
    return net, held


def benchmarks(close, vol, top_n=30, slow=50):
    rets = close.pct_change(fill_method=None)
    btc = rets["BTCUSDT"].fillna(0)
    ema_s = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    adv = vol.rolling(30, min_periods=20).mean().where(ema_s.notna())
    universe = (adv.rank(axis=1, ascending=False) <= top_n).shift(1).fillna(False)
    ew = rets.where(universe).mean(axis=1).fillna(0)  # equal-weight top-N, always invested
    return btc, ew


def stats(r, held=None):
    r = r.dropna()
    if r.empty:
        return {}
    eq = (1 + r).cumprod()
    years = len(r) / 365
    dd = eq / eq.cummax() - 1
    out = {
        "cagr": float(eq.iloc[-1] ** (1 / years) - 1) if years > 0 else None,
        "total": float(eq.iloc[-1] - 1),
        "max_dd": float(dd.min()),
        "sharpe": float(r.mean() / r.std() * np.sqrt(365)) if r.std() > 0 else None,
        "vol": float(r.std() * np.sqrt(365)),
        "best_day": float(r.max()), "worst_day": float(r.min()),
    }
    if held is not None:
        out["avg_exposure"] = float(held.sum(axis=1).loc[r.index].mean())
    return out


def periods(index):
    """Calendar chunks to check the result isn't one lucky stretch."""
    out = []
    for y0, y1 in ((2018, 2019), (2020, 2021), (2022, 2023), (2024, 2026)):
        m = (index.year >= y0) & (index.year <= y1)
        if m.sum() > 120:
            out.append((f"{y0}–{y1}", m))
    return out


# --------------------------------------------------------------------------- #
def run(params=None):
    p = {**DEFAULT, **(params or {})}
    close, vol = load_panel()
    start = close.index[close.notna().sum(axis=1) >= 10][0]  # need a real universe
    close, vol = close.loc[start:], vol.loc[start:]

    net, held = backtest(close, vol, **p)
    btc, ew = benchmarks(close, vol, top_n=p["top_n"], slow=p["slow"])
    first = net.index[p["slow"]]  # skip the slow-EMA warm-up
    net, btc, ew, held_ = net.loc[first:], btc.loc[first:], ew.loc[first:], held.loc[first:]

    # Same strategy with other EMA lengths: a real effect shouldn't hinge on one setting
    grid, grid_rets = [], {}
    for g in GRID:
        r, h = backtest(close, vol, **{**p, **g})
        grid_rets[f"ema{g['fast']}/{g['slow']}"] = r.loc[first:]
        grid.append({**g, **stats(r.loc[first:], h)})
    # ...and it shouldn't hinge on one lucky stretch of years either
    per = []
    for lab, m in periods(net.index):
        row = {"period": lab, "btc": stats(btc[m]), "equal_weight": stats(ew[m])}
        row.update({k: stats(r[m]) for k, r in grid_rets.items()})
        per.append(row)

    # Current state (latest closed day)
    w, uptrend, universe, ema_f, ema_s, sigma = signals_and_weights(close, vol, **p)
    d = close.index[-1]
    cur = []
    for sym in universe.columns[universe.loc[d].fillna(False)]:
        c = close[sym]
        up = uptrend[sym]
        # days since the trend state last changed
        changes = up.ne(up.shift()).cumsum()
        streak = int((changes == changes.loc[d]).sum())
        cur.append({
            "symbol": sym, "in_trend": bool(up.loc[d]), "days": streak, "weight": float(w.loc[d, sym]),
            "close": float(c.loc[d]), "ret_7d": float(c.loc[d] / c.shift(7).loc[d] - 1),
            "ret_30d": float(c.loc[d] / c.shift(30).loc[d] - 1),
            "vs_ema_slow": float(c.loc[d] / ema_s[sym].loc[d] - 1),
            "vol_30d": float(vol[sym].iloc[-30:].mean()), "volatility": float(sigma[sym].loc[d] * np.sqrt(365)),
        })
    cur.sort(key=lambda x: (-x["in_trend"], -x["weight"], -x["ret_30d"]))

    # Per coin: did the trend rule beat simply holding it, over its own history?
    for x in cur:
        sym = x["symbol"]
        c1 = close[[sym]].dropna()
        if len(c1) < 365:
            x["history_days"] = len(c1)
            continue
        r_t, h_t = backtest(c1, vol[[sym]].loc[c1.index], fast=p["fast"], slow=p["slow"], top_n=1, vol_days=p["vol_days"])
        r_h = c1[sym].pct_change(fill_method=None).fillna(0)
        st_t, st_h = stats(r_t.iloc[p["slow"]:]), stats(r_h.iloc[p["slow"]:])
        x.update(history_days=len(c1), trend_sharpe=st_t["sharpe"], hold_sharpe=st_h["sharpe"],
                 trend_cagr=st_t["cagr"], hold_cagr=st_h["cagr"], trend_dd=st_t["max_dd"], hold_dd=st_h["max_dd"])

    # The strongest single result: the rule applied to BTC alone
    btc_t, btc_h = backtest(close[["BTCUSDT"]], vol[["BTCUSDT"]], fast=p["fast"], slow=p["slow"], top_n=1, vol_days=p["vol_days"])
    btc_t = btc_t.loc[first:]
    for row, (lab, m) in zip(per, periods(net.index)):
        row["btc_trend"] = stats(btc_t[m])

    eq = lambda r: [[int(t.timestamp()), float(v)] for t, v in (1 + r).cumprod().items()]
    report = {
        "created": int(time.time()), "params": p, "start": str(net.index[0].date()), "end": str(net.index[-1].date()),
        "coins_in_data": int(close.shape[1]),
        "strategy": stats(net, held_), "btc": stats(btc), "equal_weight": stats(ew),
        "btc_trend": stats(btc_t, btc_h.loc[first:]),
        "grid": grid,
        "periods": per,
        "equity": {"strategy": eq(net), "btc": eq(btc), "equal_weight": eq(ew), "btc_trend": eq(btc_t)},
        "exposure": [[int(t.timestamp()), float(v)] for t, v in held_.sum(axis=1).items()],
        "current": {"date": str(d.date()), "rows": cur,
                    "invested": float(w.loc[d].sum()), "in_trend": int(sum(x["in_trend"] for x in cur)),
                    "universe": len(cur)},
    }
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report))
    return report


def print_report(r):
    line = "=" * 84
    f = lambda v: "—" if v is None else f"{v * 100:+.0f}%"
    print(f"{line}\nDAILY TREND FOLLOWING  {r['start']} → {r['end']}  ({r['coins_in_data']} coins, top {r['params']['top_n']} by volume)\n{line}")
    print(f"{'':<26}{'CAGR':>8}{'total':>10}{'max DD':>9}{'Sharpe':>8}{'exposure':>10}")
    for lab, s in (("Trend EMA%d/%d" % (r['params']['fast'], r['params']['slow']), r["strategy"]), ("Buy & hold BTC", r["btc"]), ("Equal-weight top coins", r["equal_weight"])):
        print(f"{lab:<26}{f(s['cagr']):>8}{f(s['total']):>10}{f(s['max_dd']):>9}{s['sharpe']:>8.2f}{(f(s.get('avg_exposure')) if 'avg_exposure' in s else ''):>10}")
    print("\nParameter grid (whole period):")
    for g in r["grid"]:
        print(f"  EMA{g['fast']}/{g['slow']:<4}  CAGR {f(g['cagr'])}  max DD {f(g['max_dd'])}  Sharpe {g['sharpe']:.2f}  exposure {f(g['avg_exposure'])}")
    print("\nBy period (CAGR / max DD):")
    keys = [k for k in r["periods"][0] if k != "period"]
    print(f"  {'period':<11}" + "".join(f"{k:>22}" for k in keys))
    for row in r["periods"]:
        print(f"  {row['period']:<11}" + "".join(f"{f(row[k].get('cagr')) + ' / ' + f(row[k].get('max_dd')):>22}" for k in keys))
    c = r["current"]
    print(f"\nToday ({c['date']}): {c['in_trend']} of {c['universe']} top coins in uptrend → {c['invested'] * 100:.0f}% invested")
    for x in c["rows"][:12]:
        print(f"  {x['symbol']:<12}{'TREND' if x['in_trend'] else '  —  '}  {x['days']:>4}d  weight {x['weight'] * 100:5.1f}%"
              f"  30d {x['ret_30d'] * 100:+6.1f}%  vs EMA{r['params']['slow']} {x['vs_ema_slow'] * 100:+6.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.download:
        t0 = time.time()
        n = download_all(progress=lambda d, n, s: print(f"\r{d}/{n} {s:<16}", end=""))
        print(f"\nDownloaded daily candles for {n} pairs in {time.time() - t0:.0f}s")
    print_report(run())


if __name__ == "__main__":
    main()
