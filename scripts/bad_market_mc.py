import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def run_bad_market_projection(csv_path='backtest_results/smart_pullback_equity.csv', target_year=2030, iterations=1000):
    df_eq = pd.read_csv(csv_path)
    if 'daily_return' not in df_eq.columns:
        df_eq['daily_return'] = df_eq['equity'].pct_change()
        
    daily_returns = df_eq['daily_return'].dropna().values
    
    # Segregate returns
    pos_returns = daily_returns[daily_returns > 0]
    neg_returns = daily_returns[daily_returns < 0]
    zero_returns = daily_returns[daily_returns == 0]
    
    print(f"Historical Positive Days: {len(pos_returns)}")
    print(f"Historical Negative Days: {len(neg_returns)}")
    
    # STRESS TEST 1: The "Lost Decade" (Heavy Bear Market)
    # We force the simulation to pick negative returns 50% of the time, 
    # instead of the historical win rate.
    # We also amplify negative returns by 1.2x to simulate worse slippage/crashes.
    
    stressed_neg_returns = neg_returns * 1.2  # 20% deeper losses
    
    current_equity = df_eq['equity'].iloc[-1]
    
    # 5 years * 252 days
    days_to_simulate = 5 * 252
    
    final_equities = []
    max_drawdowns = []
    all_paths = []
    
    for _ in range(iterations):
        # Build a stressed path
        # 40% chance of positive return, 50% chance of negative return, 10% flat
        choices = np.random.choice(['pos', 'neg', 'zero'], size=days_to_simulate, p=[0.40, 0.50, 0.10])
        
        sim_returns = np.zeros(days_to_simulate)
        
        pos_count = np.sum(choices == 'pos')
        neg_count = np.sum(choices == 'neg')
        zero_count = np.sum(choices == 'zero')
        
        if pos_count > 0:
            sim_returns[choices == 'pos'] = np.random.choice(pos_returns, size=pos_count, replace=True)
        if neg_count > 0:
            sim_returns[choices == 'neg'] = np.random.choice(stressed_neg_returns, size=neg_count, replace=True)
            
        sim_equity = current_equity * np.cumprod(1 + sim_returns)
        
        final_equities.append(sim_equity[-1])
        all_paths.append(sim_equity)
        
        peak = np.maximum.accumulate(np.insert(sim_equity, 0, current_equity))
        drawdown = (np.insert(sim_equity, 0, current_equity) - peak) / peak
        max_drawdowns.append(np.min(drawdown))
        
    # Plotting
    plt.figure(figsize=(12, 6))
    for path in all_paths[:100]:
        plt.plot(path, color='darkred', alpha=0.05)
        
    median_path = np.median(all_paths, axis=0)
    p5_path = np.percentile(all_paths, 5, axis=0)
    p95_path = np.percentile(all_paths, 95, axis=0)
    
    plt.plot(median_path, color='red', linewidth=2, label='Median Projection')
    plt.plot(p95_path, color='orange', linestyle='--', label='95th Percentile')
    plt.plot(p5_path, color='black', linestyle='--', label='5th Percentile')
    
    plt.title(f'STRESS TEST: "Lost Decade" Bear Market Projection to 2030', fontsize=14)
    plt.xlabel('Future Trading Days')
    plt.ylabel('Projected Equity (INR)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('backtest_results/bad_market_projection.png', dpi=300)
    plt.close()
    
    print("="*40)
    print("STRESS TEST: BAD MARKET PROJECTION")
    print("Assumptions: 50% Down Days, 40% Up Days, Losses amplified by 1.2x")
    print("="*40)
    print(f"Projected Median Equity  : ₹{np.median(final_equities):,.2f}")
    print(f"Projected 95th Percentile: ₹{np.percentile(final_equities, 95):,.2f} (Optimistic)")
    print(f"Projected 5th Percentile : ₹{np.percentile(final_equities, 5):,.2f} (Pessimistic)")
    print(f"Projected Max Drawdown   : {np.median(max_drawdowns)*100:.2f}%")
    print("="*40 + "\n")

if __name__ == "__main__":
    run_bad_market_projection()
