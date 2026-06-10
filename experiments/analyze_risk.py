import pandas as pd

df = pd.read_csv('backtest_results/trades.csv')
df['risk_pct'] = (df['entry_price'] - (df['entry_price'] / (1 + df['pnl_pct']/100))) # Not right
df['risk_pct'] = ((df['entry_price'] - (df['entry_price'] * (1 - 0.015))) / df['entry_price']) * 100 # No, the SL is ob_low * 0.985

# We didn't export initial_sl to csv. But we can estimate it for STOP_LOSS trades:
sl_trades = df[df['exit_reason'] == 'STOP_LOSS']
print(f"Average SL hit %: {sl_trades['pnl_pct'].mean():.2f}%")
print(f"Max SL hit %: {sl_trades['pnl_pct'].min():.2f}%")

