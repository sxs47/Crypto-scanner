"""
Blow-off watch (short side, paper-tracked).

Setup, checked on 4h candles built from the stored 15m candles:
    24h volume ≥ 20× the coin's normal daily volume (median of the previous 30 days)
    price up ≥ 20% over those 24h, 24h turnover ≥ $1M, first such event for the coin in 7 days
Short entry: the first 4h close ≥ 10% below the highest price since the event (within 72h).
Paper trade: target −10%, stop +10% (stop counts first inside one 15m candle), else closed after 48h.
Market phase: "weak" when an equal-weight index of liquid altcoins is below its 50-day average.

Research (Bybit, 4h, 2023–2026, ~1,500 events): about +1% per trade on average vs −0.6% for random
shorts, mostly in weak altcoin markets — but it lost money through 2024 in every market phase,
so it is tracked on paper here before anyone trusts it.
"""

import time

import numpy as np
import pandas as pd

VOL_X, RUN, MIN_TURNOVER = 20.0, 0.20, 1_000_000
COOLDOWN_BARS, WAIT_BARS = 42, 18          # 7 days, 72h (in 4h bars)
OFF_TOP, TP, SL, HOLD_H = 0.10, 0.10, 0.10, 48
FEE = 0.0011
RULE = {"vol_x": VOL_X, "run": RUN, "off_top": OFF_TOP, "tp": TP, "sl": SL, "hold_h": HOLD_H}
H4 = pd.Timedelta(hours=4)


def bars4h(df):
    """Completed 4h candles (aligned to 00/04/08… UTC) from 15m candles."""
    b = df.resample("4h", label="left", closed="left", origin="epoch").agg(
        {"high": "max", "low": "min", "close": "last", "quote_volume": "sum"})
    last_open = df["close"].last_valid_index()
    return b[b.index + pd.Timedelta(minutes=225) <= last_open]


def alt_phase(frames):
    """Daily Series: True = weak altcoin market (alt index below its 50-day average), known at that day's start."""
    closes, turn = {}, {}
    for s, df in frames.items():
        if s == "BTCUSDT":
            continue
        d = df.resample("1D").agg({"close": "last", "quote_volume": "sum"})
        closes[s], turn[s] = d["close"], d["quote_volume"]
    C, T = pd.DataFrame(closes), pd.DataFrame(turn)
    liquid = (T.rolling(7).mean() >= MIN_TURNOVER).shift(1, fill_value=False)
    ret = np.log(C / C.shift()).where(liquid)
    idx = ret.mean(axis=1).fillna(0).cumsum()
    ma = idx.rolling(50).mean()
    gap = (np.exp(idx - ma) - 1).where(ma.notna())
    p = pd.DataFrame({"weak": (idx < ma).where(ma.notna()), "gap": gap}).iloc[:-1]   # completed days only
    p.index = p.index + pd.Timedelta(days=1)       # a day's close is what's known at the start of the next day
    return p


def scan(symbol, df, phase=None):
    """All blow-off events for one coin with their paper-trade state."""
    b = bars4h(df)
    if len(b) < 6 * 36:
        return []
    h, l, c, q = (b[k].to_numpy(dtype=float) for k in ("high", "low", "close", "quote_volume"))
    v24 = pd.Series(q).rolling(6).sum()
    normal = pd.concat([v24.shift(6 * k) for k in range(1, 31)], axis=1).median(axis=1, skipna=False).to_numpy()
    v24 = v24.to_numpy()
    ratio = v24 / normal
    p24 = c / np.concatenate([np.full(6, np.nan), c[:-6]]) - 1
    n = len(b)
    out, last = [], -10**9
    for i in np.flatnonzero((ratio >= VOL_X) & (p24 >= RUN) & (v24 >= MIN_TURNOVER)):
        if i - last < COOLDOWN_BARS:
            continue
        last = i
        t_event = b.index[i] + H4
        day = t_event.floor("D")
        ph = None
        if phase is not None and day in phase.index and not pd.isna(phase.at[day, "weak"]):
            ph = bool(phase.at[day, "weak"])
        ev = {"id": f"{symbol}|{int(t_event.timestamp())}", "symbol": symbol, "event_at": int(t_event.timestamp()),
              "event_price": float(c[i]), "run_24h": float(p24[i]), "vol_x": float(ratio[i]),
              "turnover24": float(v24[i]), "weak_market": ph, "top": float(h[i]),
              "entry_at": None, "entry": None, "status": "watching", "exit_at": None, "result": None}
        for j in range(i + 1, min(n, i + WAIT_BARS + 1)):
            top = h[i: j + 1].max()
            ev["top"] = float(top)
            if c[j] <= top * (1 - OFF_TOP):
                ev.update(entry_at=int((b.index[j] + H4).timestamp()), entry=float(c[j]), status="open")
                break
        else:
            if i + WAIT_BARS < n:
                ev["status"] = "no pullback"
        if ev["status"] == "open":
            _paper(df, ev)
        out.append(ev)
    return out


def _paper(df, ev):
    """Follow the paper short on 15m candles."""
    e, t0 = ev["entry"], pd.Timestamp(ev["entry_at"], unit="s", tz="UTC")
    w = df.loc[t0: t0 + pd.Timedelta(hours=HOLD_H) - pd.Timedelta(minutes=15)].dropna(subset=["close"])
    for t, row in w.iterrows():
        if row["high"] >= e * (1 + SL):
            ev.update(status="stopped", exit_at=int((t + pd.Timedelta(minutes=15)).timestamp()), result=-SL - FEE)
            return
        if row["low"] <= e * (1 - TP):
            ev.update(status="target", exit_at=int((t + pd.Timedelta(minutes=15)).timestamp()), result=TP - FEE)
            return
    if len(w) and w.index[-1] + pd.Timedelta(minutes=15) >= t0 + pd.Timedelta(hours=HOLD_H):
        ev.update(status="time exit", exit_at=int((w.index[-1] + pd.Timedelta(minutes=15)).timestamp()),
                  result=(e - w["close"].iloc[-1]) / e - FEE)
        return
    last = df["close"].dropna()
    ev["result"] = (e - last.iloc[-1]) / e - FEE      # still open: current paper P&L


def _summary(rows):
    if not rows:
        return {"n": 0}
    r = np.array([x["result"] for x in rows])
    return {"n": len(rows), "win": float((r > 0).mean()), "avg": float(r.mean()), "total": float(r.sum())}


def scan_all(frames):
    t0 = time.time()
    phase = alt_phase(frames)
    items = [ev for s, df in frames.items() for ev in scan(s, df, phase)]
    done = [x for x in items if x["status"] in ("target", "stopped", "time exit")]
    now_phase = None
    known = phase.dropna()
    if len(known):
        p = known.iloc[-1]
        now_phase = {"weak": bool(p["weak"]), "gap": float(p["gap"]), "day": int(known.index[-1].timestamp())}
    record = {"all": _summary(done),
              "weak": _summary([x for x in done if x["weak_market"] is True]),
              "strong": _summary([x for x in done if x["weak_market"] is False]),
              "since": min((x["entry_at"] for x in done), default=None)}
    return {"items": items, "phase": now_phase, "record": record, "rule": RULE, "built": time.time(),
            "seconds": round(time.time() - t0, 1)}
