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

fetcher = DataFetcher()
data_dict = fetcher.fetch_all()
vix_df = data_dict.get('VIX')
nifty_df = data_dict.get('Nifty50')
vix_df['VIX_Multiplier'] = calculate_vix_filter(vix_df)

# Hardcode the tightened regime filter
nifty_df['EMA_20'] = nifty_df['Close'].ewm(span=20, adjust=False).mean()
nifty_df['Regime_Allowed'] = nifty_df['Close'] > nifty_df['EMA_20']

def add_techs(df):
    df = df.copy()
    df['Prev_Close'] = df['Close'].shift(1)
    df['TR'] = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Prev_Close']), abs(df['Low'] - df['Prev_Close'])))
    df['ATR'] = df['TR'].rolling(window=14).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['Uptrend'] = df['Close'] > df['EMA_50']
    
    df['Pullback_20'] = (df['Low'] <= df['EMA_20']) & (df['Close'] > df['EMA_20'])
    df['Pullback_10'] = (df['Low'] <= df['EMA_10']) & (df['Close'] > df['EMA_10'])
    
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
original_trade_init = strategy.Trade.__init__

def run_sim(risk_pct, max_pos, pullback_type, candle_req, t1_mult, t2_mult):
    os.environ['RISK_PCT'] = str(risk_pct)
    os.environ['MAX_POSITIONS'] = str(max_pos)
    
    for ticker in data_dict:
        if ticker.endswith('.NS') and ticker != 'NIFTYBEES.NS':
            df = data_dict[ticker]
            if pullback_type == 'ema10':
                df['Tech_Valid'] = df['Uptrend'] & df['Pullback_10']
            else:
                df['Tech_Valid'] = df['Uptrend'] & df['Pullback_20']
                
            if candle_req == 'strict':
                df['Entry_Signal'] = df['Tech_Valid'] & df['Candle_Signal']
            elif candle_req == 'bullish':
                df['Entry_Signal'] = df['Tech_Valid'] & df['Is_Bullish']
            else:
                df['Entry_Signal'] = df['Tech_Valid']

    def custom_pos_size(equity, r_pct, entry_price, atr, vix_multiplier):
        if pd.isna(atr) or atr <= 0 or vix_multiplier == 0: return 0, 0.0
        stop_loss = entry_price - (1.5 * atr)
        if (entry_price - stop_loss) / entry_price > 0.15: return 0, stop_loss
        risk_per_share = entry_price - stop_loss
        if risk_per_share <= 0: return 0, 0.0
        shares = int(np.floor((equity * risk_pct / risk_per_share) * vix_multiplier))
        return shares, stop_loss
    strategy.calculate_position_size = custom_pos_size

    def custom_trade_init(self, stock, entry_date, entry_price, stop_loss, target_1, target_2, size, vix_val, sector):
        rps = entry_price - stop_loss
        t1 = entry_price + (t1_mult * rps)
        t2 = entry_price + (t2_mult * rps)
        original_trade_init(self, stock, entry_date, entry_price, stop_loss, t1, t2, size, vix_val, sector)
    strategy.Trade.__init__ = custom_trade_init

    engine = BacktestEngine(data_dict, max_positions=max_pos, risk_pct=risk_pct)
    from data_fetcher import SECTOR_INDICES
    for name in SECTOR_INDICES.keys():
        if name in engine.data_dict:
            engine.data_dict[name]['Sector_RS'] = True

    engine.run()
    strategy.calculate_position_size = original_calc
    strategy.Trade.__init__ = original_trade_init
    
    res = calculate_metrics(engine.equity_curve, engine.closed_trades)
    return {
        'risk': risk_pct,
        'pos': max_pos,
        'pullback': pullback_type,
        'candle': candle_req,
        't1': t1_mult,
        't2': t2_mult,
        'cagr': float(str(res.get('CAGR (%)', '0')).replace('%', '')),
        'trades': res.get('Total Trades', 0),
        'win_rate': res.get('Win Rate (%)', '0%'),
        'dd': float(str(res.get('Max Drawdown (%)', '0')).replace('%', ''))
    }

params = {
    'risk_pct': [0.03, 0.04],
    'max_pos': [5, 8],
    'pullback_type': ['ema20', 'ema10'],
    'candle_req': ['strict', 'bullish'],
    't1_mult': [2.0],
    't2_mult': [3.0, 4.0]
}
keys, values = zip(*params.items())
experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"Running {len(experiments)} experiments...")
results = []
for p in experiments:
    r = run_sim(**p)
    results.append(r)

res_df = pd.DataFrame(results)
res_df = res_df.sort_values('cagr', ascending=False)
print("\nTop Configs by CAGR (with DD < 25%):")
good_dd = res_df[res_df['dd'] > -25] # e.g. -18 > -25
print(good_dd.head(10).to_string())

