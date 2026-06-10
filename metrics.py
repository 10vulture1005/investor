import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from engine import Trade

def calculate_metrics(equity_curve: list, closed_trades: list[Trade], risk_free_rate: float = 0.06):
    df_eq = pd.DataFrame(equity_curve)
    df_eq.set_index('date', inplace=True)
    
    initial_equity = df_eq['equity'].iloc[0]
    final_equity = df_eq['equity'].iloc[-1]
    
    days = (df_eq.index[-1] - df_eq.index[0]).days
    years = days / 365.25 if days > 0 else 1
    
    cagr = ((final_equity / initial_equity) ** (1 / years) - 1) if years > 0 else 0
    
    df_eq['peak'] = df_eq['equity'].cummax()
    df_eq['drawdown'] = (df_eq['equity'] - df_eq['peak']) / df_eq['peak']
    max_dd = df_eq['drawdown'].min()
    
    total_trades = len(closed_trades)
    
    wins = []
    losses = []
    hold_days = []
    
    for t in closed_trades:
        hold_days.append(t.hold_days)
        total_pnl = sum([e['net_pnl'] for e in t.exit_events])
        if total_pnl > 0:
            wins.append(total_pnl)
        else:
            losses.append(total_pnl)
            
    win_rate = len(wins) / total_trades if total_trades > 0 else 0
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
    
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0
    avg_win_loss = (avg_win / avg_loss) if avg_loss > 0 else float('inf')
    
    avg_hold = np.mean(hold_days) if hold_days else 0
    
    df_eq['daily_return'] = df_eq['equity'].pct_change()
    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
    excess_returns = df_eq['daily_return'] - daily_rf
    sharpe = np.sqrt(252) * (excess_returns.mean() / excess_returns.std()) if excess_returns.std() > 0 else 0
    
    metrics = {
        'Final Equity': final_equity,
        'Total Return (%)': (final_equity / initial_equity - 1) * 100,
        'CAGR (%)': cagr * 100,
        'Sharpe Ratio': sharpe,
        'Max Drawdown (%)': max_dd * 100,
        'Win Rate (%)': win_rate * 100,
        'Avg Win / Avg Loss': avg_win_loss,
        'Profit Factor': profit_factor,
        'Total Trades': total_trades,
        'Avg Hold Days': avg_hold
    }
    
    return metrics, df_eq

def monthly_returns_table(df_eq: pd.DataFrame) -> pd.DataFrame:
    if df_eq.empty or 'equity' not in df_eq.columns:
        return pd.DataFrame()
        
    df = df_eq[['equity']].copy()
    monthly = df.resample('ME').last()
    monthly['Return'] = monthly['equity'].pct_change()
    
    if len(monthly) > 0 and len(df) > 0:
        monthly.iloc[0, monthly.columns.get_loc('Return')] = (monthly['equity'].iloc[0] / df['equity'].iloc[0]) - 1
        
    monthly['Year'] = monthly.index.year
    monthly['Month'] = monthly.index.month
    
    pivot = monthly.pivot(index='Year', columns='Month', values='Return') * 100
    
    month_names = {
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
        7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
    }
    pivot.rename(columns=month_names, inplace=True)
    
    yearly = df['equity'].resample('YE').last()
    yearly_ret = yearly.pct_change()
    if len(yearly_ret) > 0:
        yearly_ret.iloc[0] = (yearly.iloc[0] / df['equity'].iloc[0]) - 1
        
    yearly_ret.index = yearly_ret.index.year
    pivot['YTD'] = yearly_ret * 100
    
    return pivot.round(2)

def plot_historical_performance(df_eq: pd.DataFrame):
    """Generates and saves the historical equity curve and drawdown chart."""
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(df_eq.index, df_eq['equity'], label='Portfolio Equity', color='#2ca02c', linewidth=2)
    plt.title('Smart Alpha 3.0: Historical Equity Curve', fontsize=14)
    plt.ylabel('Equity (INR)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.fill_between(df_eq.index, df_eq['drawdown'] * 100, 0, color='#d62728', alpha=0.3)
    plt.plot(df_eq.index, df_eq['drawdown'] * 100, color='#d62728', linewidth=1)
    plt.title('Underwater Drawdown Profile', fontsize=14)
    plt.ylabel('Drawdown (%)', fontsize=12)
    plt.xlabel('Date')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('backtest_results', exist_ok=True)
    plt.savefig('backtest_results/historical_performance.png', dpi=300)
    plt.close()

def run_monte_carlo(df_eq: pd.DataFrame, iterations: int = 1000):
    """
    Runs a Monte Carlo simulation by resampling daily returns with replacement.
    Returns median final equity and median max drawdown.
    """
    daily_returns = df_eq['daily_return'].dropna().values
    n_days = len(daily_returns)
    initial_equity = df_eq['equity'].iloc[0]
    
    final_equities = []
    max_drawdowns = []
    all_paths = []
    
    for _ in range(iterations):
        # Resample returns with replacement
        resampled_returns = np.random.choice(daily_returns, size=n_days, replace=True)
        # Reconstruct equity curve
        sim_equity = initial_equity * np.cumprod(1 + resampled_returns)
        
        final_equities.append(sim_equity[-1])
        all_paths.append(sim_equity)
        
        # Calculate drawdown for this path
        peak = np.maximum.accumulate(sim_equity)
        drawdown = (sim_equity - peak) / peak
        max_drawdowns.append(np.min(drawdown))
        
    # Plotting
    plt.figure(figsize=(12, 6))
    for path in all_paths[:100]: # Plot first 100 paths to avoid clutter
        plt.plot(path, color='blue', alpha=0.05)
    
    median_path = np.median(all_paths, axis=0)
    plt.plot(median_path, color='red', linewidth=2, label='Median Path')
    
    plt.title('Monte Carlo Historical Simulation (1000 Paths)', fontsize=14)
    plt.xlabel('Trading Days')
    plt.ylabel('Equity (INR)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('backtest_results/monte_carlo_historical.png', dpi=300)
    plt.close()
        
    return np.median(final_equities), np.median(max_drawdowns)

def run_monte_carlo_projection(df_eq: pd.DataFrame, target_year: int = 2030, iterations: int = 1000):
    """
    Projects the equity curve forward to a target year using Monte Carlo resampling.
    Returns the median, 5th percentile, and 95th percentile of the projected final equity.
    """
    daily_returns = df_eq['daily_return'].dropna().values
    current_equity = df_eq['equity'].iloc[-1]
    current_date = df_eq.index[-1]
    
    # Estimate trading days remaining (roughly 252 days per year)
    years_remaining = target_year - current_date.year
    if years_remaining <= 0:
        return current_equity, current_equity, current_equity
        
    days_to_simulate = int(years_remaining * 252)
    
    final_equities = []
    max_drawdowns = []
    all_paths = []
    
    for _ in range(iterations):
        resampled_returns = np.random.choice(daily_returns, size=days_to_simulate, replace=True)
        sim_equity = current_equity * np.cumprod(1 + resampled_returns)
        final_equities.append(sim_equity[-1])
        all_paths.append(sim_equity)
        
        peak = np.maximum.accumulate(np.insert(sim_equity, 0, current_equity))
        drawdown = (np.insert(sim_equity, 0, current_equity) - peak) / peak
        max_drawdowns.append(np.min(drawdown))
        
    # Plotting
    plt.figure(figsize=(12, 6))
    for path in all_paths[:100]:
        plt.plot(path, color='green', alpha=0.05)
        
    median_path = np.median(all_paths, axis=0)
    p5_path = np.percentile(all_paths, 5, axis=0)
    p95_path = np.percentile(all_paths, 95, axis=0)
    
    plt.plot(median_path, color='red', linewidth=2, label='Median Projection')
    plt.plot(p95_path, color='orange', linestyle='--', label='95th Percentile')
    plt.plot(p5_path, color='black', linestyle='--', label='5th Percentile')
    
    plt.title(f'Monte Carlo Forward Projection to {target_year} ({days_to_simulate} Days)', fontsize=14)
    plt.xlabel('Future Trading Days')
    plt.ylabel('Projected Equity (INR)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('backtest_results/monte_carlo_projection.png', dpi=300)
    plt.close()
        
    median_eq = np.median(final_equities)
    p5_eq = np.percentile(final_equities, 5)
    p95_eq = np.percentile(final_equities, 95)
    median_dd = np.median(max_drawdowns)
    
    return median_eq, p5_eq, p95_eq, median_dd, days_to_simulate
