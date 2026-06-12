import logging
import os
import pandas as pd

from core.data_fetcher import DataFetcher
from core.indicators import calculate_technical_indicators
from core.engine import BacktestEngine
from core.metrics import calculate_metrics, monthly_returns_table, run_monte_carlo, run_monte_carlo_projection, plot_historical_performance

# Custom symbols mapped to their Yahoo Finance tickers
CUSTOM_STOCKS = {
    'MOTHERSON': 'MOTHERSON.NS',
    'LT FOODS': 'LTFOODS.NS',
    'NALCO': 'NATIONALUM.NS',
    'MOTILAL NASDAQ 100': 'MON100.NS',
    'RECL': 'RECLTD.NS',
    'NIFTYBEES': 'NIFTYBEES.NS',
    'EXIDE INDUSTRY': 'EXIDEIND.NS',
    'NTPC': 'NTPC.NS',
    'NIFTY GOLD BEES ETF': 'GOLDBEES.NS',
    'ICICILOVOL': 'LOWVOLIETF.NS',
    'ETERNALS': 'ETERNAL.NS',
    'JK PAPER': 'JKPAPER.NS',
    'ADITYA BIRLA CAP': 'ABCAPITAL.NS',
    'NV20IETF': 'NV20IETF.NS',
    'JAIPRAKASH POWER': 'JPPOWER.NS',
    'IOCL': 'IOC.NS',
    'INDIAN OVERSEAS BANK': 'IOB.NS',
    'EIH': 'EIHOTEL.NS',
    'NCC': 'NCC.NS',
    'NHPC': 'NHPC.NS',
    'GTPL HATHWAYS': 'GTPL.NS',
    'IEX': 'IEX.NS',
    'EVEXIA LIFECARE': 'EVEXIA.BO'
}

class MyPortEngine(BacktestEngine):
    def __init__(self, data_dict, start_capital=10000000.0):
        super().__init__(data_dict, start_capital)
        # Override self.stocks to include all fetched custom stocks, excluding the indices
        self.stocks = [k for k in self.data_dict.keys() if k not in ['Nifty50', 'VIX']]

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    start_date = '2021-01-01'
    end_date = '2025-12-31'
    output_dir = 'my port back'
    
    logger.info(f"Initializing Custom Portfolio Backtest ({start_date} to {end_date})...")
    
    fetcher = DataFetcher(start_date=start_date, end_date=end_date)
    raw_data_dict = {}
    
    # Fetch Nifty50 and VIX (Engine needs them for regime filters)
    logger.info("Fetching Nifty50 and VIX...")
    raw_data_dict['Nifty50'] = fetcher.fetch_data('^NSEI')
    raw_data_dict['VIX'] = fetcher.fetch_data('^INDIAVIX')
    
    # Fetch custom stocks
    for name, ticker in CUSTOM_STOCKS.items():
        logger.info(f"Fetching {name} ({ticker})...")
        df = fetcher.fetch_data(ticker)
        if not df.empty:
            raw_data_dict[ticker] = df
        else:
            logger.warning(f"Failed to fetch data for {name} ({ticker})")

    # Precompute Indicators
    logger.info("Precomputing technical indicators...")
    data_dict = {}
    for ticker, df in raw_data_dict.items():
        if not df.empty:
            data_dict[ticker] = calculate_technical_indicators(df)
            
    # Run Engine
    logger.info("Starting engine loop...")
    if os.path.exists('models/custom_xgb_filter.pkl'):
        logger.info("⚡ Custom XGBoost ML Filter is ACTIVE. Low probability trades will be rejected.")

    # We also need to tell the engine to load the custom model.
    # The engine loads 'models/xgb_filter.pkl' by default, so we need to override the engine's model loading.
    engine = MyPortEngine(data_dict, start_capital=10000000.0)
    if os.path.exists('models/custom_xgb_filter.pkl'):
        import joblib
        engine.ml_model = joblib.load('models/custom_xgb_filter.pkl')
        logger.info("Loaded Custom XGBoost ML Filter model.")
        
    equity_curve, closed_trades = engine.run()
    
    # Calculate Metrics
    logger.info("Calculating metrics...")
    metrics, df_eq = calculate_metrics(equity_curve, closed_trades)
    
    print("\n" + "="*40)
    print("BACKTEST SUMMARY (Custom Portfolio)")
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
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot historical equity curve and drawdown
    plot_historical_performance(df_eq, output_dir=output_dir)
    
    # Monte Carlo Historical
    logger.info("Running Monte Carlo Simulation (1000 iterations)...")
    mc_eq, mc_dd = run_monte_carlo(df_eq, output_dir=output_dir)
    print("="*40)
    print("MONTE CARLO SIMULATION (1000 paths)")
    print("="*40)
    print(f"Median Final Equity      : ₹{mc_eq:,.2f}")
    print(f"Median Max Drawdown (%)  : {mc_dd*100:.2f}%")
    print("="*40 + "\n")
    
    # Monte Carlo Projection to 2030
    logger.info("Running Monte Carlo Projection to 2030...")
    proj_med, proj_p5, proj_p95, proj_dd, sim_days = run_monte_carlo_projection(df_eq, target_year=2030, output_dir=output_dir)
    print("="*40)
    print(f"FORWARD PROJECTION (Till 2030) - {sim_days} Trading Days")
    print("="*40)
    print(f"Projected Median Equity  : ₹{proj_med:,.2f}")
    print(f"Projected 95th Percentile: ₹{proj_p95:,.2f} (Optimistic)")
    print(f"Projected 5th Percentile : ₹{proj_p5:,.2f} (Pessimistic)")
    print(f"Projected Max Drawdown   : {proj_dd*100:.2f}%")
    print("="*40 + "\n")
    
    df_eq.to_csv(os.path.join(output_dir, 'custom_portfolio_equity.csv'))
    logger.info(f"Equity curve saved to {output_dir}/custom_portfolio_equity.csv")

if __name__ == "__main__":
    main()
