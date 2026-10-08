"""
Live 15m candle updates and setup alerts.

After each 15m candle closes, fetch the newly closed candle(s) for every pair,
append them to the CSVs and the in-memory frames, and run the same setup rules
as setups.py on them. Alerts use the backtest's exact rules and cooldown.
"""

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import exchange
import scanner
import setups

INTERVAL_S = 900
TAIL = 700            # candles of history used to evaluate indicators (EMA200 warm-up)
WARMUP = 300
CSV_COLS = scanner.CSV_COLUMNS
FRAME_COLS = ["open", "high", "low", "close", "quote_volume", "trades", "taker_buy_quote"]


def fetch_since(symbol, last_open):
    """Closed 15m candles after `last_open` (a UTC Timestamp), in the shared row shape."""
    now_ms = int(time.time() * 1000)
    end_ms = now_ms - now_ms % (INTERVAL_S * 1000)  # start of the current, unclosed candle
    start_ms = int(last_open.timestamp() * 1000) + INTERVAL_S * 1000
    return exchange.klines(symbol, "15m", start_ms, end_ms) if start_ms < end_ms else []


def append_rows(symbol, frame, rows, candle_dir=None):
    """Append candle rows (exchange.py row shape) to the CSV and return the extended in-memory frame."""
    raw = pd.DataFrame([r[:11] for r in rows], columns=CSV_COLS)
    times = pd.to_datetime(raw["open_time"], unit="ms", utc=True)
    csv = raw.copy()
    csv["open_time"] = times.dt.strftime("%Y-%m-%d %H:%M:%S")
    csv["close_time"] = pd.to_datetime(raw["close_time"], unit="ms", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")
    path = (candle_dir or scanner.CANDLE_DIR) / f"{symbol}.csv"
    csv.to_csv(path, mode="a", header=not path.exists(), index=False)

    new = raw[FRAME_COLS].astype(float)
    new.index = times
    new.index.name = "open_time"
    df = pd.concat([frame, new])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.reindex(pd.date_range(df.index[0], df.index[-1], freq="15min")).rename_axis("open_time")


# --------------------------------------------------------------------------- #
def alerts_for(symbol, df, since, exchange="bybit"):
    """Setup alerts on candles with open time > `since`, using the backtest's cooldown."""
    tail = df.iloc[-TAIL:]
    if len(tail) < WARMUP + 10:
        return []
    ind = setups.indicators(tail)
    masks = setups.detect(ind)
    qv24 = tail["quote_volume"].fillna(0).rolling(96).sum()
    out = []
    for setup, m in masks.items():
        m = m.copy()
        m.iloc[:WARMUP] = False
        for t in setups.apply_cooldown(m):
            if t <= since:
                continue
            r = ind.loc[t]
            out.append({
                "id": f"{exchange}|{symbol}|{setup}|{int(t.timestamp())}",
                "exchange": exchange, "symbol": symbol, "setup": setup, "time": int(t.timestamp()),
                "price": float(r["close"]),
                "vol_x": _f(r["vol_x"]), "vol_x_2h": _f(r["vol_x_2h"]), "chg": _f(r["chg"]),
                "rsi": _f(r["rsi"]), "ret_24h": _f(r["ret_24h"]),
                "ext_ema21": _f(r["close"] / r["ema21"] - 1), "macd_hist": _f(r["macd_hist"]),
                "volume_24h": _f(qv24.at[t]),
            })
    return out


def _f(x):
    x = float(x)
    return None if np.isnan(x) or np.isinf(x) else x


def outcome(df, alert):
    """How the alert has played out so far (within its first 24h)."""
    t = pd.Timestamp(alert["time"], unit="s", tz="UTC")
    entry = alert["price"]
    w = df.loc[t + pd.Timedelta(minutes=15): t + pd.Timedelta(hours=24)].dropna(subset=["close"])
    last = df["close"].dropna()
    res = {"now": float(last.iloc[-1] / entry - 1), "hours": (last.index[-1] - t).total_seconds() / 3600}
    # 1h follow-through (None = the hour after the alert isn't over yet)
    c = setups.confirmation(df, t) if t in df.index else None
    res.update(confirmed=None if c is None else c["confirmed"],
               chg_1h=None if c is None else c["chg_1h"],
               held=None if c is None else c["held"],
               higher_high=None if c is None else c["higher_high"],
               above_ema9=None if c is None else c["above_ema9"],
               confirm_time=None if c is None else int(c["time_1h"].timestamp()))
    if w.empty:
        res.update(peak=None, trough=None, reached5=False, reached10=False, done=False)
        return res
    peak, trough = float(w["high"].max() / entry - 1), float(w["low"].min() / entry - 1)
    res.update(peak=peak, trough=trough, reached5=peak >= 0.05, reached10=peak >= 0.10,
               done=len(w) >= 96, ret_24h=float(w["close"].iloc[-1] / entry - 1) if len(w) >= 96 else None)
    return res
