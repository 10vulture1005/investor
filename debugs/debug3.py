from data_fetcher import DataFetcher, SECTOR_MAP, SECTOR_INDICES
from filters import calculate_vix_filter, calculate_sector_rs_filter, calculate_fii_proxy_filter, identify_order_block
from strategy import generate_signals

fetcher = DataFetcher()
data = fetcher.fetch_all()

vix_df = data.get('VIX')
vix_df['VIX_Multiplier'] = calculate_vix_filter(vix_df)

niftybees_df = data.get('NIFTYBEES')
niftybees_df['FII_Allowed'] = calculate_fii_proxy_filter(niftybees_df)

nifty_df = data.get('^NSEI')

for sec in SECTOR_INDICES.values():
    if sec in data:
        data[sec]['Sector_RS'] = calculate_sector_rs_filter(data[sec], nifty_df)

stock = 'RELIANCE.NS'
df = data[stock]
df_weekly = fetcher.get_weekly_data(df)
df_smc = identify_order_block(df, df_weekly)
df_signals = generate_signals(df_smc)

print(f"Total Raw Signals for {stock}: {df_signals.get('Entry_Signal', []).sum()}")

# Trace each signal
if 'Entry_Signal' in df_signals:
    for date, row in df_signals[df_signals['Entry_Signal']].iterrows():
        # Get previous date (the date the signal was actually evaluated for tomorrow's open)
        prev_idx = df_signals.index.get_loc(date)
        
        vix_mult = vix_df.loc[date].get('VIX_Multiplier', 1.0) if date in vix_df.index else 1.0
        fii_allow = niftybees_df.loc[date].get('FII_Allowed', False) if date in niftybees_df.index else False
        
        sector = SECTOR_MAP.get(stock)
        sector_ticker = SECTOR_INDICES.get(sector)
        sec_df = data.get(sector_ticker)
        sec_allow = sec_df.loc[date].get('Sector_RS', False) if sec_df is not None and date in sec_df.index else False
        
        print(f"Date: {date.date()} | VIX Mult: {vix_mult} | FII: {fii_allow} | Sector RS: {sec_allow}")

