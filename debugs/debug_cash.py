from backtest_engine import BacktestEngine
from data_fetcher import DataFetcher
import os

os.environ['INITIAL_CAPITAL'] = '1000000'
os.environ['MAX_POSITIONS'] = '5'
os.environ['RISK_PCT'] = '0.05'

fetcher = DataFetcher()
data_dict = fetcher.fetch_all()

from main import calculate_vix_filter, calculate_market_regime_filter, calculate_technical_indicators, generate_signals
import strategy

# Setup
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

for i in range(1, len(eq_curve)):
    prev_cash = eq_curve[i-1]['cash']
    curr_cash = eq_curve[i]['cash']
    diff = curr_cash - prev_cash
    if diff < -100000 or diff > 100000:
        print(f"{eq_curve[i]['date'].date()} Cash changed by: {diff:,.2f} -> Current Cash: {curr_cash:,.2f}")

print(f"Final Cash: {eq_curve[-1]['cash']}")

