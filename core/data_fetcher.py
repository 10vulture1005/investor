# pyrefly: ignore [missing-import]
import yfinance as yf
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
NIFTY_50_TICKERS = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS',
    'ITC.NS', 'SBIN.NS', 'LT.NS', 'BHARTIARTL.NS', 'BAJFINANCE.NS',
    'HINDUNILVR.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'KOTAKBANK.NS', 'TITAN.NS',
    'MARUTI.NS', 'SUNPHARMA.NS', 'ULTRACEMCO.NS', 'TRENT.NS', 'HCLTECH.NS',
    'TATASTEEL.NS', 'NTPC.NS', 'WIPRO.NS', 'BAJAJFINSV.NS', 'M&M.NS',
    'POWERGRID.NS', 'NESTLEIND.NS', 'ADANIENT.NS', 'GRASIM.NS', 'TECHM.NS',
    'INDUSINDBK.NS', 'HINDALCO.NS', 'ONGC.NS', 'JSWSTEEL.NS', 'BRITANNIA.NS',
    'CIPLA.NS', 'EICHERMOT.NS', 'ADANIPORTS.NS', 'HEROMOTOCO.NS', 'APOLLOHOSP.NS',
    'DIVISLAB.NS', 'COALINDIA.NS', 'DRREDDY.NS', 'HDFCLIFE.NS', 'SBILIFE.NS',
    'UPL.NS', 'BAJAJ-AUTO.NS', 'TATACHEM.NS', 'TATACONSUM.NS', 'BPCL.NS' # Using TATACHEM and BPCL to fill 50, standard proxies.
]

# Note: In real scenarios, these symbols change over time. 
# We're using a static representative list for the 2020-2024 backtest period.

SECTOR_INDICES = {
    'Bank': '^NSEBANK',
    'IT': '^CNXIT',
    'Pharma': '^CNXPHARMA',
    'Auto': '^CNXAUTO',
    'FMCG': '^CNXFMCG',
    'Nifty50': '^NSEI'
}

# Mapping Nifty 50 stocks to their primary sector (only the 5 sectors are mapped for the strategy)
SECTOR_MAP = {
    'HDFCBANK.NS': 'Bank', 'ICICIBANK.NS': 'Bank', 'SBIN.NS': 'Bank', 'AXISBANK.NS': 'Bank', 'KOTAKBANK.NS': 'Bank', 'INDUSINDBK.NS': 'Bank',
    'TCS.NS': 'IT', 'INFY.NS': 'IT', 'HCLTECH.NS': 'IT', 'WIPRO.NS': 'IT', 'TECHM.NS': 'IT',
    'SUNPHARMA.NS': 'Pharma', 'CIPLA.NS': 'Pharma', 'APOLLOHOSP.NS': 'Pharma', 'DIVISLAB.NS': 'Pharma', 'DRREDDY.NS': 'Pharma',
    'MARUTI.NS': 'Auto', 'TATAMOTORS.NS': 'Auto', 'M&M.NS': 'Auto', 'EICHERMOT.NS': 'Auto', 'HEROMOTOCO.NS': 'Auto', 'BAJAJ-AUTO.NS': 'Auto',
    'ITC.NS': 'FMCG', 'HINDUNILVR.NS': 'FMCG', 'NESTLEIND.NS': 'FMCG', 'BRITANNIA.NS': 'FMCG', 'TATACONSUM.NS': 'FMCG'
}

class DataFetcher:
    def __init__(self, start_date='2020-01-01', end_date='2024-12-31', cache_dir='data_cache'):
        self.start_date = start_date
        self.end_date = end_date
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_filepath(self, ticker, interval='1d'):
        clean_ticker = ticker.replace('^', '').replace('.', '_')
        return self.cache_dir / f"{clean_ticker}_{interval}_{self.start_date}_{self.end_date}.parquet"

    def fetch_data(self, ticker, interval='1d'):
        """Fetches data from Yahoo Finance and caches it as parquet."""
        filepath = self._get_filepath(ticker, interval)
        if filepath.exists():
            return pd.read_parquet(filepath)
            
        logger.info(f"Downloading {ticker} [{interval}] from {self.start_date} to {self.end_date}")
        
        try:
            # Note: yfinance auto_adjust=True is the default for history(), handles corporate actions.
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(start=self.start_date, end=self.end_date, interval=interval, auto_adjust=True)
            
            if df.empty:
                logger.warning(f"No data found for {ticker}")
                return pd.DataFrame()
                
            # Clean up index
            df.index = pd.to_datetime(df.index).tz_localize(None) # Remove timezone for easier processing
            
            # Save to cache
            df.to_parquet(filepath)
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {str(e)}")
            return pd.DataFrame()

    def get_weekly_data(self, df_daily):
        """Resamples daily data to weekly OHLCV."""
        if df_daily.empty:
            return pd.DataFrame()
            
        # Define weekly aggregation logic
        logic = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }
        
        # Resample starting on Monday, ending on Friday ('W-FRI')
        df_weekly = df_daily.resample('W-FRI').agg(logic)
        df_weekly.dropna(inplace=True)
        return df_weekly

    def fetch_all(self):
        """Fetches all required data and returns a dictionary of DataFrames."""
        data_dict = {}
        
        # 1. Fetch Nifty 50 constituents
        for ticker in NIFTY_50_TICKERS:
            df = self.fetch_data(ticker)
            if not df.empty:
                data_dict[ticker] = df
                
        # 2. Fetch VIX
        data_dict['VIX'] = self.fetch_data('^INDIAVIX')
        
        # 3. Fetch Niftybees for FII Proxy
        data_dict['NIFTYBEES'] = self.fetch_data('NIFTYBEES.NS')
        
        # 4. Fetch Sector Indices
        for name, ticker in SECTOR_INDICES.items():
            data_dict[name] = self.fetch_data(ticker)
            
        logger.info("All data fetching complete.")
        return data_dict

if __name__ == "__main__":
    fetcher = DataFetcher(start_date='2020-01-01', end_date='2020-02-01') # Short period for test
    data = fetcher.fetch_all()
    print("Keys fetched:", data.keys())
