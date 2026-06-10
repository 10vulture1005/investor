import pandas as pd
import numpy as np
import os
import itertools
import logging
from backtest_engine import BacktestEngine
from report import calculate_metrics
import strategy
from data_fetcher import DataFetcher
from filters import calculate_vix_filter, calculate_market_regime_filter

logging.getLogger().setLevel(logging.CRITICAL)

os.environ['INITIAL_CAPITAL'] = '1000000'
os.environ['MAX_POSITIONS'] = '5'

fetcher = DataFetcher()
data_dict = fetcher.fetch_all()
vix_df = data_dict.get('VIX')
nifty_df = data_dict.get('Nifty50')
vix_df['VIX_Multiplier'] = calculate_vix_filter(vix_df)
nifty_df['Regime_Allowed'] = calculate_market_regime_filter(nifty_df)

def add_techs(df):
    df = df.copy()
    df['Prev_Close'] = df['Close'].shift(1)
    df['TR'] = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Prev_Close']), abs(df['Low'] - df['Prev_Close'])))
    df['ATR'] = df['TR'].rolling(window=14).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['Uptrend'] = df['Close'] > df['EMA_50']
    
    df['Is_Bullish'] = df['Close'] > df['Open']
    df['Prev_Bearish'] = (df['Close'].shift(1) < df['Open'].shift(1))
    df['Bullish_Engulfing'] = df['Is_Bullish'] & df['Prev_Bearish'] & (df['Open'] <= df['Close'].shift(1)) & (df['Close'] >= df['Open'].shift(1))
    df['Body_Size'] = abs(df['Close'] - df['Open'])
    df['Lower_Wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['Pin_Bar'] = df['Lower_Wick'] >= (2 * df['Body_Size'])
    df['Candle_Signal'] = df['Bullish_Engulfing'] | df['Pin_Bar']
    return df

for ticker in data_dict:
    if ticker.endswith('.NS') and ticker != 'NIFTYBEES.NS':
        data_dict[ticker] = add_techs(data_dict[ticker])

original_calc = strategy.calculate_position_size

def run_sim(rsi_thresh, atr_mult, risk_limit, risk_pct):
    os.environ['RISK_PCT'] = str(risk_pct)
    
    for ticker in data_dict:
        if ticker.endswith('.NS') and ticker != 'NIFTYBEES.NS':
            df = data_dict[ticker]
            df['Tech_Valid'] = df['Uptrend'] & (df['RSI'] < rsi_thresh)
            df['Entry_Signal'] = df['Tech_Valid'] & df['Candle_Signal']

    def custom_pos_size(equity, r_pct, entry_price, atr, vix_multiplier):
        if pd.isna(atr) or atr <= 0 or vix_multiplier == 0: return 0, 0.0
        stop_loss = entry_price - (atr_mult * atr)
        if (entry_price - stop_loss) / entry_price > risk_limit: return 0, stop_loss
        risk_per_share = entry_price - stop_loss
        if risk_per_share <= 0: return 0, 0.0
        shares = int(np.floor((equity * risk_pct / risk_per_share) * vix_multiplier))
        return shares, stop_loss
    strategy.calculate_position_size = custom_pos_size

    engine = BacktestEngine(data_dict, risk_pct=risk_pct)
    # Turn off Sector RS logic in engine dynamically
    original_run = engine.run
    from data_fetcher import SECTOR_INDICES
    for name in SECTOR_INDICES.keys():
        if name in engine.data_dict:
            engine.data_dict[name]['Sector_RS'] = True

    engine.run()
    strategy.calculate_position_size = original_calc
    
    res = calculate_metrics(engine.equity_curve, engine.closed_trades)
    return {
        'rsi': rsi_thresh,
        'atr': atr_mult,
        'limit': risk_limit,
        'risk_pct': risk_pct,
        'cagr': float(str(res.get('CAGR (%)', '0')).replace('%', '')),
        'total': res.get('Total Return (%)', '0%'),
        'trades': res.get('Total Trades', 0),
        'win_rate': res.get('Win Rate (%)', '0%'),
        'dd': res.get('Max Drawdown (%)', '0%')
    }

params = {
    'rsi': [65, 70],
    'atr': [1.5, 2.0],
    'limit': [0.15],
    'risk_pct': [0.05, 0.08, 0.10]
}
keys, values = zip(*params.items())
experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

results = []
for p in experiments:
    r = run_sim(p['rsi'], p['atr'], p['limit'], p['risk_pct'])
    results.append(r)

res_df = pd.DataFrame(results)
res_df = res_df.sort_values('cagr', ascending=False)
print("\nTop Configs:")
print(res_df.head(10).to_string())

