from data_fetcher import DataFetcher, SECTOR_INDICES
from filters import calculate_fii_proxy_filter, calculate_sector_rs_filter

fetcher = DataFetcher()
data = fetcher.fetch_all()

nifty = data.get('^NSEI')
niftybees = data.get('NIFTYBEES')
niftybees['FII_Allowed'] = calculate_fii_proxy_filter(niftybees)

print("FII_Allowed counts:")
print(niftybees['FII_Allowed'].value_counts(dropna=False))

print("\nSample FII:")
print(niftybees[['Close', 'Open', 'Volume', 'FII_Allowed']].tail(10))

bank = data.get('^NSEBANK')
bank['Sector_RS'] = calculate_sector_rs_filter(bank, nifty)
print("\nSector_RS counts:")
print(bank['Sector_RS'].value_counts(dropna=False))

