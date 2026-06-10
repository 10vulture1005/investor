import logging
import os
from dotenv import load_dotenv
from data_fetcher import DataFetcher
from filters import calculate_vix_filter, calculate_sector_rs_filter, calculate_fii_proxy_filter, identify_order_block
from strategy import generate_signals
from backtest_engine import BacktestEngine
from report import calculate_metrics, generate_report

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('main')

def main():
    logger.info("Initializing Quant Backtest Engine...")
    
    # Get config from env
    start_date = os.getenv('START_DATE', '2020-01-01')
    end_date = os.getenv('END_DATE', '2024-12-31')
    initial_capital = float(os.getenv('INITIAL_CAPITAL', '1000000.0'))
    max_positions = int(os.getenv('MAX_POSITIONS', '5'))
    risk_pct = float(os.getenv('RISK_PCT', '0.015'))
    risk_free_rate = float(os.getenv('RISK_FREE_RATE', '0.065'))
    
    # 1. Fetch Data
    fetcher = DataFetcher(start_date=start_date, end_date=end_date)
    data_dict = fetcher.fetch_all()
    
    # Quick sanity check
    if '^INDIAVIX' not in data_dict and 'VIX' not in data_dict:
        logger.error("VIX data missing. Aborting.")
        return
        
    vix_df = data_dict.get('VIX')
    nifty_df = data_dict.get('^NSEI')
    niftybees_df = data_dict.get('NIFTYBEES')
    
    logger.info("Applying Market-wide Filters...")
    # 2. Market-wide Filters
    vix_df['VIX_Multiplier'] = calculate_vix_filter(vix_df)
    niftybees_df['FII_Allowed'] = calculate_fii_proxy_filter(niftybees_df)
    
    # 3. Sector RS
    sector_tickers = ['^NSEBANK', '^CNXIT', '^CNXPHARMA', '^CNXAUTO', '^CNXFMCG']
    for sec in sector_tickers:
        if sec in data_dict:
            data_dict[sec]['Sector_RS'] = calculate_sector_rs_filter(data_dict[sec], nifty_df)
            
    # 4. Stock-specific Filters & Signals
    logger.info("Generating signals for individual stocks...")
    for ticker, df in data_dict.items():
        if ticker.endswith('.NS') and ticker != 'NIFTYBEES.NS':
            # Create weekly data
            df_weekly = fetcher.get_weekly_data(df)
            
            # Identify Order Block (SMC Filter)
            df_smc = identify_order_block(df, df_weekly)
            
            # Generate Signals (Candlestick patterns within OB)
            df_signals = generate_signals(df_smc)
            
            data_dict[ticker] = df_signals
            
    # 5. Run Backtest
    engine = BacktestEngine(
        data_dict=data_dict,
        start_capital=initial_capital,
        max_positions=max_positions,
        risk_pct=risk_pct
    )
    
    equity_curve, closed_trades = engine.run()
    
    # 6. Generate Report
    metrics = calculate_metrics(equity_curve, closed_trades, initial_capital=initial_capital, risk_free_rate=risk_free_rate)
    generate_report(metrics, closed_trades, output_dir='backtest_results')
    
    logger.info("Pipeline Execution Finished.")

if __name__ == "__main__":
    main()
