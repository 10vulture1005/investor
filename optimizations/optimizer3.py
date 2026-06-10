import pandas as pd
import numpy as np
import os
from data_fetcher import DataFetcher
from filters import calculate_vix_filter, calculate_sector_rs_filter, calculate_market_regime_filter

os.environ['INITIAL_CAPITAL'] = '1000000'
os.environ['MAX_POSITIONS'] = '5'
os.environ['RISK_PCT'] = '0.05'

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
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + (2 * df['BB_Std'])
    df['BB_Lower'] = df['BB_Mid'] - (2 * df['BB_Std'])
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

from backtest_engine import BacktestEngine
from report import calculate_metrics
import strategy
original_calc = strategy.calculate_position_size

def run_sim(rsi_thresh, atr_mult, use_sector, bb_touch, risk_limit):
    for ticker in data_dict:
        if ticker.endswith('.NS') and ticker != 'NIFTYBEES.NS':
            df = data_dict[ticker]
            if bb_touch:
                df['Tech_Valid'] = df['Uptrend'] & (df['Low'] <= df['BB_Lower']) & (df['RSI'] < rsi_thresh)
            else:
                df['Tech_Valid'] = df['Uptrend'] & (df['RSI'] < rsi_thresh)
            df['Entry_Signal'] = df['Tech_Valid'] & df['Candle_Signal']

    def custom_pos_size(equity, risk_pct, entry_price, atr, vix_multiplier):
        if pd.isna(atr) or atr <= 0 or vix_multiplier == 0: return 0, 0.0
        stop_loss = entry_price - (atr_mult * atr)
        if (entry_price - stop_loss) / entry_price > risk_limit: return 0, stop_loss
        risk_per_share = entry_price - stop_loss
        if risk_per_share <= 0: return 0, 0.0
        shares = int(np.floor((equity * risk_pct / risk_per_share) * vix_multiplier))
        return shares, stop_loss
    strategy.calculate_position_size = custom_pos_size

    import logging
    logging.getLogger().setLevel(logging.CRITICAL)

    engine = BacktestEngine(data_dict)
    # Turn off Sector RS logic in engine dynamically
    if not use_sector:
        original_run = engine.run
        # Monkey patch the loop... well, let's just make the columns all True if False
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
        'sector': use_sector,
        'bb': bb_touch,
        'cagr': res.get('CAGR (%)', 0),
        'total': res.get('Total Return (%)', '0%'),
        'trades': res.get('Total Trades', 0),
        'win_rate': res.get('Win Rate (%)', '0%')
    }

results = []
# Try a combo that is likely to generate lots of good trades
r = run_sim(rsi_thresh=60, atr_mult=2.0, use_sector=False, bb_touch=False, risk_limit=0.15)
results.append(r)
r = run_sim(rsi_thresh=55, atr_mult=2.0, use_sector=False, bb_touch=False, risk_limit=0.15)
results.append(r)

print(pd.DataFrame(results))
