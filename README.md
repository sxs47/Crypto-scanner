# Pump Scanner

A local web dashboard that scans Bybit USDT perpetual futures (crypto only) on
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
| **Alerts** | **Blow-off watch** (paper shorts after 20× volume run-ups, with an altcoin-market switch and live track record); **Pre-pump watch** (volume and candle size jumped 8×/3× while price is flat; "squeeze setup" when most accounts are short) with its 30-day track record; live 15m setups (Trend / Ignition), 1-hour follow-through check, EMA/MACD charts, futures context (open interest, funding, long/short), notifications |
| **Early** | One list of coins with fresh early signals (last 24h): ATR_SM 4h Buy (confirmed or blinking), Run pick, pre-pump volume, Radar — scored by how many line up, with a BTC 4h warning. Telegram "Early entries" alert when a coin reaches 2+ (or 3+) signals. |
| *(Early, ATR)* **Spot / futures** | spotflow.py: for coins with Bybit spot, spot share of 24h turnover vs its 30-day normal; ⚠️ *futures-led* (< 0.8×) signals faded more often in tests (281 coins, 2023–26). Spot 4h turnover kept 45 days in data/spot_4h/. |
| **Paper** | paper.py: two fake $500 accounts on live Bybit prices — *Plan* (filtered confirmed ATR 4h Buys, 10% margin 3x, stop −5%, half at +3% then break-even, rest +8%, max 24h) and *Fast* (15m EMA7 over 21/50/200, 10% margin 5x, stop −3%, exit EMA7<EMA21 or +8%, max 24h). Fees included, funding not; state in data/paper.json. |
| **Market** | market.py: total / BTC / altcoin market cap and BTC dominance (CoinGecko, snapshots in data/market/), Fear & Greed (alternative.me), Bybit funding / open interest / long-short, and BTC·ETH·SOL technicals on 1d/4h/1h/15m (EMA 7/21/50/200, RSI, MACD, volume, ATR_SM). |
| **ATR** | Every Bybit USDT perpetual (no volume minimum), ATR_SM 5/5 on 4h: Buy and Sell flips — *blinking* (the open candle would flip at the live price) and *confirmed* (candle closed with the flip). |
| *(Alerts)* **ATR_SM Buy · 4h** | atrstop.py: the ATR_SM_BuySell TradingView indicator (ATR trailing line, by dysrupt / nifty_alchemist) on 4h, Period 5 / Multiplier 5 — best of 63 settings on 15m/1h/4h; Buys paper-tracked (+10% before −5%, +20% before −10%, best before Sell). Run picks show whether ATR_SM was in Buy mode. |
| **Trend** | Daily EMA20/50 trend rule with volatility-target position sizes: today's status per coin and a backtest since 2020 |
| **Planner** | Risk-based position sizing for Bybit futures: size, leverage, liquidation price, fees, funding, plus a risk-of-ruin simulation |
| **Journal** | Your closed trades (Bybit "Closed P&L" CSV import or manual entry) with win rate, expectancy and P&L by coin / hour / weekday / side; stored in `data/journal.json`, not in git |
| **Radar** | A model (radar.py, trained on ~24,000 pump candles since 2023) flags the top 0.5% of coins most likely to make a big move in the next 24h, every 4h, with the reasons and an honest live track record. Big move up OR down, not a buy signal. Retrain: `python radar.py --train` |
| *(Radar)* **▲ Run picks** | runpicks.py: ~173 TradingView-style indicators (4h + daily) + volume + Bybit positioning (long/short ratio, funding, open interest) → coins most likely to reach +15% before −15%; paper-tracked as +10%-before-−5%, +20%-before-−10% and best gain before a −5%/−10% stop. Retrain: `python runpicks.py --train` |
| **Carry** | Funding carry (buy on spot + short the perpetual, collect funding): every liquid coin's funding now / 7-day / 30-day as % per year, the weekly top-5 rotation picks, last 4 weeks of the rotation, and a calculator (split, liquidation distance, expected funding, fees). carry.py |
| *(everywhere)* | `new Nd` tag on coins listed on Bybit in the last 30 days (most new listings underperform BTC) |
| *(every chart)* | **Why is it moving?** button: latest news headlines (Google News RSS), Bybit announcements and CoinGecko trending for the coin (news.py, no keys). 🔥 marks coins trending on CoinGecko; the collector saves the trending list every 15 min to `data/social/` |

## Run it

Requires Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt      # Windows
# .venv/bin/pip install -r requirements.txt        # Linux / macOS

# First time only: download market data (~5 minutes in total)
.venv\Scripts\python scanner.py --days 90          # 15m candles (also fetches the coin list)
.venv\Scripts\python trend.py --download           # daily candles since 2020

.venv\Scripts\python app.py                        # then open http://127.0.0.1:5000
```

Keep `app.py` running: it fetches each new 15-minute candle, raises alerts,
and refreshes the daily trend data after every daily close.

## Scripts

| File | Purpose |
|---|---|
| `exchange.py` | Bybit data source: coin list (crypto perpetuals, ≥ $250K/day) and candles |
| `scanner.py` | Download 15m candles; label 30%+ pumps (CLI summary) |
| `signals.py` | Pre-pump indicator research + model (`python signals.py`, `--live`) |
| `setups.py` | Trend/Ignition rules, 1h confirmation, backtest |
| `live.py` | Live candle updates and alert detection used by the app |
| `trend.py` | Daily trend rule: download, backtest, current status |
| `futures.py` | Bybit futures context (open interest, funding, long/short) |
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

- Data: Bybit USDT perpetual futures. Stocks, ETFs, commodities and forex
  (Bybit `symbolType`), stablecoins and gold are excluded; coins in Bybit's
  Innovation zone are kept and tagged. Bybit's USDT perpetuals start in 2020.
- Only coins listed today are in the data (survivor bias); this flatters every
  long-only backtest.
- Do not expose the app to the internet without adding a login: the API can
  pause scanning and start downloads.
