import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def calculate_metrics(equity_curve, closed_trades, initial_capital=1000000.0, risk_free_rate=0.065):
    """Calculates backtest performance metrics."""
    if not equity_curve:
        return {}
        
    df_eq = pd.DataFrame(equity_curve)
    df_eq.set_index('date', inplace=True)
    
    # 1. Total Return (%)
    final_equity = df_eq['equity'].iloc[-1]
    total_return = (final_equity / initial_capital) - 1.0
    
    # Daily returns
    df_eq['daily_return'] = df_eq['equity'].pct_change()
    
    # 2. CAGR (%)
    days = (df_eq.index[-1] - df_eq.index[0]).days
    years = days / 365.25
    cagr = ((final_equity / initial_capital) ** (1/years) - 1.0) if years > 0 else 0
    
    # 3. Sharpe Ratio
    annualized_vol = df_eq['daily_return'].std() * np.sqrt(252)
    sharpe = (cagr - risk_free_rate) / annualized_vol if annualized_vol > 0 else 0
    
    # 4. Sortino Ratio
    downside_returns = df_eq['daily_return'][df_eq['daily_return'] < 0]
    downside_vol = downside_returns.std() * np.sqrt(252)
    sortino = (cagr - risk_free_rate) / downside_vol if downside_vol > 0 else 0
    
    # 5. Max Drawdown
    df_eq['peak'] = df_eq['equity'].cummax()
    df_eq['drawdown'] = (df_eq['equity'] - df_eq['peak']) / df_eq['peak']
    max_dd = df_eq['drawdown'].min()
    
    # Trade Metrics
    num_trades = len(closed_trades)
    
    if num_trades == 0:
        win_rate = 0
        avg_win_loss = 0
        profit_factor = 0
        avg_hold = 0
        best_trade = 0
        worst_trade = 0
    else:
        # Reconstruct full trade PnL from events
        trade_pnls = []
        hold_days = []
        for t in closed_trades:
            net_pnl = sum([e.get('net_pnl', e['pnl']) for e in t.exit_events])
            trade_pnls.append(net_pnl)
            hold_days.append(t.hold_days)
            
        trade_pnls = np.array(trade_pnls)
        wins = trade_pnls[trade_pnls > 0]
        losses = trade_pnls[trade_pnls <= 0]
        
        # 6. Win Rate
        win_rate = len(wins) / num_trades if num_trades > 0 else 0
        
        # 7. Average Win / Average Loss ratio
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
        avg_win_loss = avg_win / avg_loss if avg_loss > 0 else float('inf')
        
        # 8. Profit Factor
        gross_profit = wins.sum()
        gross_loss = abs(losses.sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # 10. Average holding period
        avg_hold = np.mean(hold_days)
        
        # 11. Best/Worst Trade
        best_trade = trade_pnls.max()
        worst_trade = trade_pnls.min()
        
    return {
        'Total Return (%)': total_return * 100,
        'CAGR (%)': cagr * 100,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown (%)': max_dd * 100,
        'Win Rate (%)': win_rate * 100,
        'Avg Win / Avg Loss': avg_win_loss,
        'Profit Factor': profit_factor,
        'Total Trades': num_trades,
        'Avg Hold Days': avg_hold,
        'Best Trade (INR)': best_trade,
        'Worst Trade (INR)': worst_trade,
        'df_eq': df_eq # used for plotting
    }

def generate_report(metrics, closed_trades, output_dir='backtest_results'):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    if not metrics:
        logger.warning("No metrics to report.")
        return
        
    df_eq = metrics.pop('df_eq')
    
    # 1. Print Summary
    print("\n" + "="*40)
    print("BACKTEST SUMMARY")
    print("="*40)
    for k, v in metrics.items():
        if 'INR' in k:
            print(f"{k:<25}: ₹{v:,.2f}")
        elif 'Ratio' in k or 'Factor' in k or 'Loss' in k:
            print(f"{k:<25}: {v:.2f}")
        elif 'Trades' in k or 'Days' in k:
            print(f"{k:<25}: {v:.0f}")
        else:
            print(f"{k:<25}: {v:.2f}%")
    print("="*40 + "\n")
    
    # 2. Export Trades CSV
    trade_data = []
    for t in closed_trades:
        net_pnl = sum([e.get('net_pnl', e['pnl']) for e in t.exit_events])
        pnl_pct = (net_pnl / (t.initial_size * t.entry_price)) * 100 if t.initial_size > 0 else 0
        reasons = "+".join(set([e['reason'] for e in t.exit_events]))
        exit_date = t.exit_events[-1]['date'] if t.exit_events else t.entry_date
        exit_price = sum([e['price']*e['size'] for e in t.exit_events]) / t.initial_size if t.initial_size > 0 else t.entry_price
        
        trade_data.append({
            'entry_date': t.entry_date,
            'exit_date': exit_date,
            'stock': t.stock,
            'entry_price': t.entry_price,
            'exit_price': exit_price,
            'pnl_pct': pnl_pct,
            'pnl_inr': net_pnl,
            'hold_days': t.hold_days,
            'exit_reason': reasons,
            'vix_at_entry': t.vix_at_entry,
            'sector': t.sector
        })
        
    df_trades = pd.DataFrame(trade_data)
    if not df_trades.empty:
        df_trades.to_csv(out_path / 'trades.csv', index=False)
        
    # --- Visualizations ---
    sns.set_theme(style="darkgrid")
    
    # V1. Equity Curve
    plt.figure(figsize=(10, 5))
    plt.plot(df_eq.index, df_eq['equity'], label='Equity', color='blue')
    plt.title('Portfolio Equity Curve')
    plt.ylabel('Equity (INR)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path / 'equity_curve.png')
    plt.close()
    
    # V2. Drawdown Curve
    plt.figure(figsize=(10, 5))
    plt.fill_between(df_eq.index, df_eq['drawdown']*100, 0, color='red', alpha=0.3)
    plt.title('Drawdown (%)')
    plt.ylabel('Drawdown (%)')
    plt.tight_layout()
    plt.savefig(out_path / 'drawdown.png')
    plt.close()
    
    # V3. Monthly Returns Heatmap
    df_eq['year'] = df_eq.index.year
    df_eq['month'] = df_eq.index.month
    monthly_ret = df_eq.groupby(['year', 'month'])['equity'].last().pct_change().unstack()
    # Handle first month logic roughly (comparing to initial capital)
    first_year = df_eq.index[0].year
    first_month = df_eq.index[0].month
    monthly_ret.loc[first_year, first_month] = (df_eq.loc[(df_eq.index.year==first_year)&(df_eq.index.month==first_month), 'equity'].iloc[-1] / metrics.get('Total Return (%)', 0)) # approx
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(monthly_ret * 100, annot=True, fmt=".1f", cmap='RdYlGn', center=0)
    plt.title('Monthly Returns (%)')
    plt.tight_layout()
    plt.savefig(out_path / 'monthly_heatmap.png')
    plt.close()
    
    # V4. Win/Loss Distribution
    if not df_trades.empty:
        plt.figure(figsize=(8, 5))
        sns.histplot(df_trades['pnl_inr'], bins=20, kde=True, 
                     hue=(df_trades['pnl_inr'] > 0), palette={True:'green', False:'red'},
                     legend=False)
        plt.title('Win/Loss Distribution (INR)')
        plt.xlabel('PnL (INR)')
        plt.tight_layout()
        plt.savefig(out_path / 'win_loss_dist.png')
        plt.close()
        
        # V5. Trade-by-trade scatter
        plt.figure(figsize=(10, 5))
        plt.scatter(df_trades.index, df_trades['pnl_pct'], 
                    c=np.where(df_trades['pnl_pct'] > 0, 'green', 'red'), alpha=0.6)
        plt.axhline(0, color='black', ls='--')
        plt.title('Trade by Trade Returns (%)')
        plt.xlabel('Trade Number')
        plt.ylabel('Return (%)')
        plt.tight_layout()
        plt.savefig(out_path / 'trade_scatter.png')
        plt.close()
        
    logger.info(f"Reports saved to {out_path}/")
