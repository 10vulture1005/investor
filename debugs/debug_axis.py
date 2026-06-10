import pandas as pd
from data_fetcher import DataFetcher
from filters import calculate_technical_indicators

fetcher = DataFetcher()
df = fetcher.fetch_all()['AXISBANK.NS']
df = calculate_technical_indicators(df)

row = df.loc['2024-01-23']
print(f"2024-01-23 Close: {row['Close']}, ATR: {row['ATR']}")
row2 = df.loc['2024-01-24']
print(f"2024-01-24 Open: {row2['Open']}, Low: {row2['Low']}")
