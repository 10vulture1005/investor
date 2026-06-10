import pandas as pd

df = pd.read_csv('backtest_results/trades.csv')

print(f"Total Trades: {len(df)}")
print("\nExit Reasons:")
print(df['exit_reason'].value_counts())

print("\nAverage PnL by Sector:")
print(df.groupby('sector')['pnl_pct'].mean())

print("\nAverage Hold Days:")
print(df.groupby('exit_reason')['hold_days'].mean())

wins = df[df['pnl_inr'] > 0]
losses = df[df['pnl_inr'] <= 0]
print(f"\nWins: {len(wins)}, Losses: {len(losses)}")
print(f"Avg Win: {wins['pnl_pct'].mean():.2f}%, Avg Loss: {losses['pnl_pct'].mean():.2f}%")

# Analyze Stop Loss distance
# A common issue is SL is hit on gap downs
print("\nStats for STOP_LOSS trades:")
print(losses[losses['exit_reason'].str.contains('STOP_LOSS')]['pnl_pct'].describe())

