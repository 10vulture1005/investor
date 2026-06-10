You are an expert Python quant developer. Build a complete backtest engine 
for the following swing trading strategy targeting Indian NSE stocks.

---

### DATA LAYER

- Use `yfinance` to fetch OHLCV data for a watchlist of NSE stocks 
  (append `.NS` suffix, e.g. `RELIANCE.NS`)
- Default watchlist: Nifty 50 constituents
- Fetch daily + weekly OHLCV data
- Fetch India VIX data (ticker: `^INDIAVIX` on yfinance)
- Backtest period: Jan 2020 to Dec 2024
- Resample daily data to weekly where needed

---

### STRATEGY RULES — implement ALL of these exactly

**Filter 1 — VIX Regime Filter**
- If India VIX > 20: halve position size
- If India VIX > 25: no new entries allowed
- If India VIX < 14: full position size allowed

**Filter 2 — Sector Relative Strength**
- Use 5 NSE sector proxies via yfinance:
  - Nifty Bank: `^NSEBANK`
  - Nifty IT: `^CNXIT`
  - Nifty Pharma: `^CNXPHARMA`
  - Nifty Auto: `^CNXAUTO`
  - Nifty FMCG: `^CNXFMCG`
- For each stock, determine its sector
- Only allow long entries if the stock's sector index has 
  outperformed Nifty 50 (`^NSEI`) over the last 10 trading days
  (i.e., sector_return_10d > nifty_return_10d)

**Filter 3 — FII Proxy (simulate with institutional flow proxy)**
- Since live FII data isn't available historically, use this proxy:
  - Calculate 5-day rolling net flow proxy = 
    (close - open) × volume for Nifty 50 ETF (`NIFTYBEES.NS`)
  - If the 5-day sum is negative (3+ of last 5 days were net sell days): 
    skip long entries

**Filter 4 — SMC Structure (simplified)**
- Weekly timeframe: price must be above 50-period EMA (uptrend bias)
- Identify the last "Order Block" on daily: 
  the most recent bearish candle body just before a bullish BOS 
  (BOS = price closes above previous 20-day swing high)
- The Order Block zone = [low of that bearish candle, high of that bearish candle]
- Current price must be within or touching this zone

---

### ENTRY TRIGGER

On the daily chart, when ALL 4 filters are green:
- Entry condition: a bullish engulfing candle OR 
  a pin bar (lower wick ≥ 2× body) forms within the Order Block zone
- Entry price: next day's open
- Maximum 5 open positions simultaneously

---

### EXIT RULES

For each trade:
- Stop Loss: 1.5% below the Order Block low
- Target 1: Exit 50% of position at 1.5× SL distance (1.5R)
- Target 2: Exit remaining 50% at 2.5× SL distance (2.5R)
- After Target 1 hit: move stop to breakeven
- Maximum hold period: 20 trading days (time stop)

---

### POSITION SIZING

- Risk 1.5% of current portfolio equity per trade
- Position size = (equity × 0.015) / (entry_price - stop_loss_price)
- Adjust for VIX filter (halve size if VIX > 20)
- Starting capital: ₹10,00,000 (10 lakhs)

---

### BACKTEST ENGINE REQUIREMENTS

- Implement using vectorbt or backtrader or a custom pandas engine
- Avoid look-ahead bias strictly (use .shift(1) where needed)
- Account for:
  - Brokerage: 0.03% per side
  - STT: 0.1% on sell side (equity delivery)
  - Slippage: 0.1% per trade

---

### OUTPUT METRICS — calculate and display all of these

1. Total Return (%)
2. CAGR (%)
3. Sharpe Ratio (annualized, risk-free = 6.5% for India)
4. Sortino Ratio
5. Max Drawdown (%)
6. Win Rate (%)
7. Average Win / Average Loss ratio
8. Profit Factor
9. Total number of trades
10. Average holding period (days)
11. Best trade / Worst trade

---

### VISUALIZATIONS — generate all using matplotlib or plotly

1. Equity curve (portfolio value over time)
2. Drawdown curve
3. Monthly returns heatmap (like a calendar)
4. Win/Loss distribution histogram
5. Trade-by-trade scatter plot (entry → exit with P&L color coded)

---

### OUTPUT FORMAT

- Print a clean summary table of all metrics
- Save all charts to a /backtest_results/ folder
- Export all trades to a CSV: 
  columns = [date, stock, entry_price, exit_price, pnl_pct, pnl_inr, 
  hold_days, exit_reason, vix_at_entry, sector]

---

### NOTES

- Handle missing data, delisted stocks, and dividend adjustments gracefully
- All timestamps in IST
- Use yfinance `auto_adjust=True` for corporate action adjustments
- Print progress logs as it runs each stock

Produce clean, modular, well-commented Python code. 
Split into: data_fetcher.py, filters.py, strategy.py, backtest_engine.py, 
report.py, and a main.py that orchestrates everything.