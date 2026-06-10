from backtest_engine import BacktestEngine
from data_fetcher import DataFetcher
import os

os.environ['INITIAL_CAPITAL'] = '1000000'
os.environ['MAX_POSITIONS'] = '5'
os.environ['RISK_PCT'] = '0.05'

fetcher = DataFetcher()
data_dict = fetcher.fetch_all()

from main import calculate_vix_filter, calculate_market_regime_filter, calculate_technical_indicators, generate_signals

vix_df = data_dict.get('VIX')
nifty_df = data_dict.get('Nifty50')
vix_df['VIX_Multiplier'] = calculate_vix_filter(vix_df)
nifty_df['Regime_Allowed'] = calculate_market_regime_filter(nifty_df)

for ticker in data_dict:
    if ticker.endswith('.NS') and ticker != 'NIFTYBEES.NS':
        df = data_dict[ticker]
        df_tech = calculate_technical_indicators(df)
        df_signals = generate_signals(df_tech)
        data_dict[ticker] = df_signals

engine = BacktestEngine(data_dict)
eq_curve, closed_trades = engine.run()

print(f"Number of closed trades: {len(closed_trades)}")
print(f"Number of open trades at end: {len(engine.open_trades)}")
for t in engine.open_trades:
    print(f"Open Trade: {t.stock} Entry: {t.entry_date.date()} Hold: {t.hold_days} Size: {t.current_size}/{t.initial_size}")

