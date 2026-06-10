import pandas as pd
from data_fetcher import DataFetcher
from filters import calculate_vix_filter, calculate_sector_rs_filter, calculate_fii_proxy_filter, identify_order_block
from strategy import generate_signals

fetcher = DataFetcher()
data = fetcher.fetch_all()
nifty = data.get('^NSEI')

total_signals = 0
for ticker, df in data.items():
    if ticker.endswith('.NS') and ticker != 'NIFTYBEES.NS':
        df_weekly = fetcher.get_weekly_data(df)
        df_smc = identify_order_block(df, df_weekly)
        df_signals = generate_signals(df_smc)
        if 'Entry_Signal' in df_signals:
            signals = df_signals['Entry_Signal'].sum()
            total_signals += signals
            if signals > 0:
                print(f"{ticker}: {signals} signals")

print(f"Total Raw Signals: {total_signals}")
