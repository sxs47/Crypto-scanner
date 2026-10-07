# Pump Scanner

A local web dashboard that scans Binance (and Bybit-only) USDT crypto pairs on
15-minute candles, records every 30%+ move, flags live volume/MACD/EMA setups,
and tests every idea against history before it goes on the page.

**Research tool, not financial advice.** The backtests on the site show that the
15-minute alerts find coins that are moving, but do not pick direction better
than chance after fees. The daily trend rule on BTC is the one result with an
edge in the data (mainly smaller drawdowns).

## Tabs

| Tab | What it shows |
|---|---|
| **Pumps** | Every +30%-in-24h move over the last ~90 days, with charts |
| **Signals** | Model-scored volatility watchlist and its out-of-sample backtest |
| **Alerts** | Live 15m setups (Trend / Ignition) on Binance coins, 1-hour follow-through check, EMA/MACD charts, futures context, notifications |
| **Bybit** | The same alerts for coins listed on Bybit but not Binance |
| **Trend** | Daily EMA20/50 trend rule: today's status per coin and a backtest since 2018 |

## Run it

Requires Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt      # Windows
# .venv/bin/pip install -r requirements.txt        # Linux / macOS

# First time only: download market data (~5 minutes in total)
.venv\Scripts\python scanner.py --days 90          # Binance 15m candles
.venv\Scripts\python bybit.py --download           # Bybit-only coins
.venv\Scripts\python trend.py --download           # daily candles since 2018

.venv\Scripts\python app.py                        # then open http://127.0.0.1:5000
```

Keep `app.py` running: it fetches each new 15-minute candle, raises alerts,
and refreshes the daily trend data after every daily close.

## Scripts

| File | Purpose |
|---|---|
| `scanner.py` | Download Binance 15m candles; label 30%+ pumps (CLI summary) |
| `signals.py` | Pre-pump indicator research + model (`python signals.py`, `--live`) |
| `setups.py` | Trend/Ignition rules, 1h confirmation, backtest (`--bybit` for Bybit coins) |
| `live.py` | Live candle updates and alert detection used by the app |
| `trend.py` | Daily trend rule: download, backtest, current status |
| `futures.py` | Binance perpetual futures context (open interest, funding, long/short) |
| `bybit.py` | Bybit-only coin list and candles |
| `app.py` | Flask server for the dashboard |
| `static/index.html` | The dashboard |

Data lives in `data/` (not in git). Model and backtest reports live in `models/`
and can be rebuilt with `signals.py`, `setups.py` and `trend.py`.

## Login, Telegram, hosting

```bash
.venv\Scripts\python app.py --set-password      # require a password (stored hashed in data/)
.venv\Scripts\python app.py --setup-telegram    # alerts to your phone via a Telegram bot
```

To run it 24/7 on a free Oracle Cloud server, follow [DEPLOY.md](DEPLOY.md).

## Notes

- Tokenized stocks/ETFs (Binance trading group `TRD_GRP_261`, Bybit `xstocks`),
  stablecoins, gold and staked/wrapped tokens are excluded.
- Only coins listed today are in the data (survivor bias); this flatters every
  long-only backtest.
- Do not expose the app to the internet without adding a login: the API can
  pause scanning and start downloads.
