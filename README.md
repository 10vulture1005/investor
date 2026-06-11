# Smart Alpha 3.0: Nifty 50 Mean Reversion Engine

Smart Alpha 3.0 is a highly concentrated, event-driven quantitative swing trading engine built for the Nifty 50 universe. It is designed to capture rapid mean-reversion bounces in high-momentum stocks, achieving a massive **~25% CAGR** without utilizing margin leverage.

## 📊 Performance Summary (ML Enhanced)

| Metric | Result |
| :--- | :--- |
| **Final Equity** | ₹106,179,925.68 (Starting: ₹10M) |
| **Total Return** | 961.80% |
| **CAGR** | **30.05%** |
| **Sharpe Ratio** | 1.44 |
| **Max Drawdown** | **-7.67%** |
| **Win Rate** | **94.37%** |
| **Profit Factor** | 44.23 |
| **Avg Win / Loss** | 2.64 |
| **Total Trades** | 71 |
| **Avg Hold Days** | 2.32 Days |

### 🔮 Monte Carlo Forward Projection (Till 2030)
Based on 1,000 resampled paths over the remaining 1,260 trading days of the decade:
* **Projected Median Equity**: ₹40.14 Crore
* **Projected 95th Percentile**: ₹69.21 Crore (Optimistic)
* **Projected 5th Percentile**: ₹24.29 Crore (Pessimistic)
* **Projected Max Drawdown**: -7.36%

---

## 🧠 Strategy Architecture

### 1. The Universe & Regime Filters
* **Universe**: Nifty 50 constituents (large-cap, highly liquid).
* **Macro Crash Filter**: The engine halts all buying if the broader market breadth is terrible (`< 30%` of Nifty 50 stocks above their 50-day SMA) OR if the Nifty 50 index is trading below its 100-day SMA.

### 2. Signal Generation (Cross-Sectional Momentum + Mean Reversion)
Rather than trading every pullback, the system ranks the entire universe daily:
* **Ranking Factor**: "Smooth Momentum" -> `ROC(90-day) / ATR(20-day)`.
* **Entry Trigger**: The engine isolates the absolute #1 ranked momentum stock, and buys it at the Open if it experiences a deep oversold pullback (`RSI(3) ≤ 30` or price drops below the Lower Keltner Channel).

### 3. Hyper-Concentrated Risk Management
* **Max Positions**: 1 (The engine only holds the single highest-conviction stock).
* **Average Margin Taken**: **~100% of Equity (1x Leverage)**. 
* **Leverage Rules**: The strategy operates with a strict `target_volatility` of 8%, meaning the math attempts to allocate large sizes. However, it is strictly capped by a cash constraint. It utilizes **no borrowed margin**, deploying 100% of available cash into the single best setup.
* **Stop Loss**: Initial hard stop at `Entry - 2.0 * ATR(14)`.

### 4. High-Velocity Exits
Because mean reversion bounces are fast but short-lived, the strategy turns over capital rapidly (avg hold time: 2.6 days).
* **Primary Target**: Sells immediately at the Close if `RSI(3) > 80` or if the price crosses back above the 10-day SMA.
* **Time Stop**: Sells automatically after 10 days if the trade flatlines, freeing up capital for the next opportunity.

---

## 🚀 Running the Engine

### Installation
Ensure you have `pandas`, `numpy`, and `matplotlib` installed:
```bash
pip install pandas numpy matplotlib
```

### Execution
Run the backtest engine. It will automatically fetch data, run the historical backtest, execute the Monte Carlo simulations (both historical and forward-projected), and save the equity curves.
```bash
python main.py
```

### Visualizations
The engine will automatically generate the following visualization charts in the `backtest_results/` directory:
1. `historical_performance.png`: The historical equity curve and underwater drawdown chart.
2. `monte_carlo_historical.png`: A spaghetti plot of 1,000 historical alternate realities.
3. `monte_carlo_projection.png`: A forward-looking fan chart projecting the strategy out to the year 2030.
