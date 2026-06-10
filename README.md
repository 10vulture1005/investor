# Investor - NSE Swing Trading Backtest Engine

A fully automated, end-to-end quantitative backtesting engine designed specifically for the Indian stock market (NSE). This engine evaluates a sophisticated swing trading strategy combining Smart Money Concepts (SMC), institutional flow proxies, volatility regimes, and sector relative strength.

Built from scratch in pure Python and Pandas to ensure a zero-lookahead bias, day-by-day event loop that accurately mirrors live portfolio behavior.

## Strategy Overview

The strategy strictly applies the following filters before considering any long entries:
1. **VIX Regime Filter**: Adjusts position sizing based on `^INDIAVIX`. Halves size if VIX > 20; prohibits new entries if VIX > 25.
2. **Sector Relative Strength**: Compares 10-day returns of the stock's sector index against the Nifty 50 (`^NSEI`). Longs are only permitted if the sector is outperforming.
3. **FII Institutional Flow Proxy**: Uses a 5-day rolling net flow approximation on `NIFTYBEES.NS`. Long entries are blocked if the market is experiencing sustained net selling.
4. **SMC Structure**: Ensures the stock is above the 50-week EMA. Identifies the most recent daily "Order Block" (bearish candle preceding a breakout) and requires the current price to retest this zone.

**Trigger & Execution:**
- Looks for Bullish Engulfing or Pin Bar candlestick patterns within the established Order Block.
- Max 5 concurrent positions.
- Dynamic position sizing based on a fixed 1.5% risk of *current* fluctuating portfolio equity.
- Take Profit scaling (50% at 1.5R, 50% at 2.5R) and a 20-day time stop.

---

## Optimized Backtest Results (2017-2025)

The strategy was heavily tested and optimized across an 8-year dataset, successfully navigating the 2020 Covid-19 Black Swan crash and the choppy 2022-2024 regimes using dynamic `EMA-200` macro-crash trailing stops and flexible multi-day pullback logic.

```
Total Return (%)         : 340.93%
CAGR (%)                 : 17.94%
Sharpe Ratio             : 0.59
Sortino Ratio            : 0.82
Max Drawdown (%)         : -22.53%
Win Rate (%)             : 38.71%
Avg Win / Avg Loss       : 2.98
Profit Factor            : 1.88
Total Trades             : 186
Avg Hold Days            : 9
Best Trade (INR)         : ₹540,249.63
Worst Trade (INR)        : ₹-109,293.95
```

### Bull Market Performance (2022-2024)

During purely bullish or sideways regimes, the strategy exhibits massive outperformance with a highly asymmetrical Profit Factor:

```
Total Return (%)         : 175.81%
CAGR (%)                 : 40.40%
Sharpe Ratio             : 1.88
Sortino Ratio            : 2.70
Max Drawdown (%)         : -11.55%
Win Rate (%)             : 46.00%
Avg Win / Avg Loss       : 8.70
Profit Factor            : 7.41
Total Trades             : 50
Avg Hold Days            : 9
Best Trade (INR)         : ₹316,533.93
Worst Trade (INR)        : ₹-82,103.80
```

---
## Installation

**1. Clone the repository:**
```bash
git clone https://github.com/10vulture1005/investor.git
cd investor
```

**2. Set up the virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

---

## Configuration

The engine is highly configurable via environment variables.

**1. Create a `.env` file:**
```bash
cp .env.example .env
```

**2. Customize parameters in `.env`:**
```env
# Time horizon
START_DATE=2020-01-01
END_DATE=2024-12-31

# Portfolio rules
INITIAL_CAPITAL=1000000.0
MAX_POSITIONS=5
RISK_PCT=0.015
```

---

## Running the Backtest

To execute the engine across the Nifty 50 constituents, run:
```bash
python main.py
```

### Outputs & Reporting
The script will automatically download the historical OHLCV data from Yahoo Finance (`yfinance`) and cache it locally in `/data_cache/` using the `.parquet` format for lightning-fast subsequent runs.

Upon completion, results are exported to the `/backtest_results/` directory:
- **`trades.csv`**: A granular ledger of every executed trade, including entry/exit prices, reasons, hold days, and PnL.
- **`equity_curve.png`**: Visualizes the growth of the portfolio equity over time.
- **`drawdown.png`**: Maps the peak-to-trough declines.
- **`monthly_heatmap.png`**: A calendar view of month-by-month returns.
- **`win_loss_dist.png` & `trade_scatter.png`**: Visualizes the distribution and chronologic progression of trade outcomes.

---

## Technical Architecture

The engine is highly modular. Each file is responsible for a very specific part of the quantitative pipeline:

### 1. `main.py` (The Orchestrator)
The manager that coordinates the entire process. It loads the `.env` configuration, triggers data fetching, applies filters, runs the backtest engine, and finally requests the report generation.

### 2. `data_fetcher.py` (The Data Layer)
Communicates with the Yahoo Finance API (`yfinance`). To prevent rate-limiting, it implements a highly efficient caching system that saves downloaded data as compressed `.parquet` files in `/data_cache/`. It also handles resampling daily data into weekly data.

### 3. `filters.py` (The Constraint Logic)
Takes raw data and calculates boolean flags for our strategy rules:
- **`calculate_vix_filter`**: Analyzes the VIX regime to restrict or adjust sizing.
- **`calculate_sector_rs_filter`**: Compares sector returns against the broader Nifty 50.
- **`calculate_fii_proxy_filter`**: Calculates a 5-day rolling net flow proxy using NIFTYBEES ETF.
- **`identify_order_block`**: Calculates the 50-week EMA, scans the daily chart for bearish candles preceding breakouts (BOS), and defines the exact price range of the Order Block.

### 4. `strategy.py` (The Trigger & Sizing Logic)
Dictates exact entry timings and sizing metrics.
- **`generate_signals`**: Scans for specific candlestick patterns (Bullish Engulfing / Pin Bars) strictly inside Order Block zones.
- **`calculate_position_size`**: Determines exact share quantities based on the 1.5% portfolio risk rule, adjusting for the VIX multiplier.
- **`Trade` Class**: Tracks individual open positions, target levels, and time-stops, returning realized PnL upon exit.

### 5. `backtest_engine.py` (The Simulator)
The core simulation loop. Rather than using vectorization (which is prone to lookahead bias and struggles with portfolio constraints), it uses an **event-driven loop**. It steps through time day-by-day, updates portfolio equity based on open positions, triggers exits, and buys new generated signals based strictly on the available cash and margin of that specific day.

### 6. `report.py` (The Analytics Engine)
Crunches the final numbers. It reconstructs the equity curve, calculates standard Wall Street KPIs (CAGR, Sharpe, Sortino, Max Drawdown), and uses `matplotlib`/`seaborn` to render the visual plots and the raw `trades.csv` ledger.
