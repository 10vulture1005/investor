# Nifty 50 Swing Trading Strategy — Current State

This document explains the current mechanics of our automated trading strategy, how it selects trades, manages risk, and handles exits.

## 1. Market Regime & Macro Filters
Before evaluating any individual stock, the engine checks the overall market conditions.
- **VIX Filter**: The India VIX must be below 28. If VIX is > 22, we halve our position sizes to reduce risk during volatile periods. If VIX is > 28, we take no new trades.
- **Macro Trend (Nifty 50)**: The Nifty 50 must be trading above its 200-day Simple Moving Average (SMA). We do not buy dips in a structural bear market. If the Nifty falls 5% below its 200 SMA, a "Macro Crash" event is triggered, and all existing stop losses are aggressively moved to breakeven to protect capital.

## 2. Entry Conditions
For an individual stock in the Nifty 50 to generate a buy signal, it must pass a strict set of technical criteria:
1. **Structural Uptrend**: The stock must be trading above its 50-day EMA, above its 200-day SMA, AND the 50-day EMA must be above the 200-day SMA.
2. **The Pullback**: The stock must pull back and touch its 20-day EMA (Low <= 20 EMA) but close above it (Close > 20 EMA), showing support is holding.
3. **Bullish Conviction Candle**: The entry day must form a strong bullish candle where the close is in the upper 40% of the day's high-low range (Close Position >= 0.6).

## 3. Position Sizing & Risk Management
We use dynamic, volatility-adjusted position sizing to ensure consistent risk across all trades.
- **Capital at Risk**: We risk a maximum of 4% of total equity per trade.
- **Stop Loss**: The stop loss is placed at `Entry Price - (1.5 * ATR)`. We use Average True Range (ATR) so that volatile stocks get wider stops and quiet stocks get tighter stops.
- **Hard Cap**: If the calculated stop loss is more than 15% away from the entry price, the trade is skipped entirely (too volatile).

## 4. Trade Management & Exits
Once in a trade, we manage it dynamically to secure profits and let winners run.
- **Target 1 (T1) - 2R**: When the stock reaches a profit equal to 2x our initial risk, we sell **40% of the position**. 
- **Breakeven Stop**: Once T1 is hit, the stop loss for the remaining 60% of the position is moved to the entry price (a "free trade").
- **Trailing Stop (2x ATR)**: After T1 is hit, we begin trailing the stop loss for the remaining shares at `Highest Price - (2.0 * ATR)`. As the stock trends higher, the stop ratchets up, capturing the trend.
- **Target 2 (T2) - 3.5R**: If the stock reaches 3.5x our initial risk, we close the entire remaining position.
- **Time Stop**: If the trade has been held for 30 trading days and hasn't hit its targets or stop loss, we exit at the market close to free up capital.

---

## Current Performance Summary
*(Tested from Jan 2017 to Dec 2025)*
- **CAGR**: 17.53%
- **Win Rate**: 35.84%
- **Max Drawdown**: -28.32%
- **Profit Factor**: 1.71
- **Avg Win / Avg Loss**: 3.06

---

## 🚀 Upgrade Prompt: Reaching the Next Level

While the current engine is stable and highly profitable (17.5% CAGR with a 3.06 Reward-to-Risk ratio), it falls short of our aggressive targets. 

**Our Targets:**
- Win Rate: > 50%
- Max Drawdown: ≤ 22%
- CAGR: > 27%

*To achieve this, we need to fundamentally shift the strategy mechanics. Since you are open to any quant strategy (excluding HFT), how would you like to proceed?*

**Option A: The High-Probability Mean Reversion approach**
Switch from trend-following pullbacks to aggressive mean-reversion (e.g., buying when RSI(3) crashes below 15 in an uptrend, exiting when it bounces). *Pros: Very high win rate (60%+). Cons: Lower average win, requires heavy leverage/frequent trading to hit 27% CAGR.*

**Option B: The Breakout & Trend Following approach**
Abandon the "pullback" entry and buy 50-day or Donchian Channel breakouts with aggressive trailing stops (e.g., classic Turtle Trading). *Pros: Massive CAGR potential (>30%). Cons: Win rates usually stay around 35-45%, though we can optimize with ADX.*

**Option C: The Multi-Factor Machine Learning / Ranking approach**
Keep the pullback framework, but instead of binary rules, score every Nifty 50 stock daily based on Momentum (ROC), Volatility (ATR), and Value (Mean deviation), and always hold the top 5 ranked stocks.

**Option D: Your Custom Idea**
Do you have a specific indicator combination or quant framework you want to build?
