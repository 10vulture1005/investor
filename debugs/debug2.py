from data_fetcher import DataFetcher
from filters import calculate_vix_filter, calculate_fii_proxy_filter

fetcher = DataFetcher()
data = fetcher.fetch_all()
vix = data.get('^INDIAVIX')
niftybees = data.get('NIFTYBEES')

vix_mult = calculate_vix_filter(vix)
fii = calculate_fii_proxy_filter(niftybees)

print(f"Total days: {len(vix)}")
print(f"VIX Allowed (>0): {(vix_mult > 0).sum()}")
print(f"FII Allowed: {fii.sum()}")
print(f"Both Allowed: ((vix_mult > 0) & fii).sum()")
