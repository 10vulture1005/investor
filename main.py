import logging
import os
import pandas as pd
from dotenv import load_dotenv

from data_fetcher import DataFetcher
from indicators import calculate_technical_indicators
from engine import BacktestEngine
from metrics import calculate_metrics, monthly_returns_table, run_monte_carlo, run_monte_carlo_projection, plot_historical_performance

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    load_dotenv()
    
    start_date = os.getenv('START_DATE', '2017-01-01')
    end_date = os.getenv('END_DATE', '2025-12-31')
    
    logger.info("Initializing Smart Pullback 2.0 Backtest Engine...")
    
    # 1. Fetch Data
    fetcher = DataFetcher(start_date=start_date, end_date=end_date)
    raw_data_dict = fetcher.fetch_all()
    
    if not raw_data_dict:
        logger.error("No data fetched. Exiting.")
        return
        
    # 2. Precompute Indicators
    logger.info("Precomputing technical indicators...")
    data_dict = {}
    for ticker, df in raw_data_dict.items():
        if ticker in ['VIX', 'Nifty50']:
            # For VIX and Nifty50, we also calculate indicators just in case (Nifty needs SMA200, EMA50)
            data_dict[ticker] = calculate_technical_indicators(df)
        elif ticker.endswith('.NS') and ticker != 'NIFTYBEES.NS':
            data_dict[ticker] = calculate_technical_indicators(df)
            
    # 3. Run Engine
    logger.info("Starting engine loop...")
    engine = BacktestEngine(data_dict, start_capital=10000000.0)
    equity_curve, closed_trades = engine.run()
    
    # 4. Calculate Metrics
    logger.info("Calculating metrics...")
    metrics, df_eq = calculate_metrics(equity_curve, closed_trades)
    
    print("\n" + "="*40)
    print("BACKTEST SUMMARY (Smart Pullback 2.0)")
    print("="*40)
    for k, v in metrics.items():
        if 'INR' in k or k == 'Final Equity':
            print(f"{k:<25}: ₹{v:,.2f}")
        elif '%' in k:
            print(f"{k:<25}: {v:.2f}%")
        elif 'Ratio' in k or 'Factor' in k or 'Avg' in k:
            print(f"{k:<25}: {v:.2f}")
        else:
            print(f"{k:<25}: {v}")
    print("="*40 + "\n")
    
    print("MONTHLY RETURNS (%)")
    print("-" * 60)
    monthly_table = monthly_returns_table(df_eq)
    print(monthly_table.to_string())
    print("-" * 60 + "\n")
    
    # Plot historical equity curve and drawdown
    plot_historical_performance(df_eq)
    
    # Monte Carlo Historical
    logger.info("Running Monte Carlo Simulation (1000 iterations)...")
    mc_eq, mc_dd = run_monte_carlo(df_eq)
    print("="*40)
    print("MONTE CARLO SIMULATION (1000 paths)")
    print("="*40)
    print(f"Median Final Equity      : ₹{mc_eq:,.2f}")
    print(f"Median Max Drawdown (%)  : {mc_dd*100:.2f}%")
    print("="*40 + "\n")
    
    # Monte Carlo Projection to 2030
    logger.info("Running Monte Carlo Projection to 2030...")
    proj_med, proj_p5, proj_p95, proj_dd, sim_days = run_monte_carlo_projection(df_eq, target_year=2030)
    print("="*40)
    print(f"FORWARD PROJECTION (Till 2030) - {sim_days} Trading Days")
    print("="*40)
    print(f"Projected Median Equity  : ₹{proj_med:,.2f}")
    print(f"Projected 95th Percentile: ₹{proj_p95:,.2f} (Optimistic)")
    print(f"Projected 5th Percentile : ₹{proj_p5:,.2f} (Pessimistic)")
    print(f"Projected Max Drawdown   : {proj_dd*100:.2f}%")
    print("="*40 + "\n")
    
    # 5. Save Equity Curve
    os.makedirs('backtest_results', exist_ok=True)
    df_eq.to_csv('backtest_results/smart_pullback_equity.csv')
    logger.info("Equity curve saved to backtest_results/smart_pullback_equity.csv")

if __name__ == "__main__":
    main()
