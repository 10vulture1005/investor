import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.metrics import run_monte_carlo_projection

def run_50_year_mc():
    df_eq = pd.read_csv('backtest_results/smart_pullback_equity.csv', index_col='date', parse_dates=True)
    if 'daily_return' not in df_eq.columns:
        df_eq['daily_return'] = df_eq['equity'].pct_change()
        
    print("Running 50-Year Monte Carlo Projection (1000 paths)...")
    # target_year=2076 ensures 50+ years from the 2025 end date.
    median_eq, p5_eq, p95_eq, median_dd, sim_days = run_monte_carlo_projection(df_eq, target_year=2076, iterations=1000)
    
    # Let's save a specific image name to not overwrite the default projection
    # Actually, the original function hardcodes the save path as 'backtest_results/monte_carlo_projection.png'
    # We can rename it right after.
    os.rename('backtest_results/monte_carlo_projection.png', 'backtest_results/mc_50_years.png')
    
    print("="*40)
    print(f"50-YEAR FORWARD PROJECTION - {sim_days} Trading Days")
    print("="*40)
    print(f"Projected Median Equity  : ₹{median_eq:,.2f}")
    print(f"Projected 95th Percentile: ₹{p95_eq:,.2f} (Optimistic)")
    print(f"Projected 5th Percentile : ₹{p5_eq:,.2f} (Pessimistic)")
    print(f"Projected Max Drawdown   : {median_dd*100:.2f}%")
    print("="*40 + "\n")

if __name__ == "__main__":
    run_50_year_mc()
