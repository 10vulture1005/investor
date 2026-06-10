import pandas as pd
import numpy as np
from data_fetcher import DataFetcher
from filters import calculate_vix_filter, calculate_sector_rs_filter, calculate_market_regime_filter
import itertools

fetcher = DataFetcher()
data_dict = fetcher.fetch_all()

vix_df = data_dict.get('VIX')
nifty_df = data_dict.get('Nifty50')

vix_df['VIX_Multiplier'] = calculate_vix_filter(vix_df)
nifty_df['Regime_Allowed'] = calculate_market_regime_filter(nifty_df)

# Precalculate basic technicals without param dependency
def add_techs(df):
    df = df.copy()
    # ATR
    df['Prev_Close'] = df['Close'].shift(1)
    df['TR'] = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Prev_Close']), abs(df['Low'] - df['Prev_Close'])))
    df['ATR'] = df['TR'].rolling(window=14).mean()
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    # BB
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + (2 * df['BB_Std'])
    df['BB_Lower'] = df['BB_Mid'] - (2 * df['BB_Std'])
    # Trend
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['Uptrend'] = df['Close'] > df['EMA_50']
    
    # Candlesticks
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

# We need to test params.
# We will just patch strategy.py's functions dynamically or run a custom loop.
from backtest_engine import BacktestEngine
from report import calculate_metrics

def run_sim(rsi_thresh, atr_mult, use_sector, bb_touch):
    # Apply Sector RS
    from data_fetcher import SECTOR_INDICES
    for name in SECTOR_INDICES.keys():
        if name in data_dict:
            if use_sector:
                data_dict[name]['Sector_RS'] = calculate_sector_rs_filter(data_dict[name], nifty_df)
            else:
                data_dict[name]['Sector_RS'] = True # Always true
                
    # Apply entry signals
    for ticker in data_dict:
        if ticker.endswith('.NS') and ticker != 'NIFTYBEES.NS':
            df = data_dict[ticker]
            if bb_touch:
                df['Tech_Valid'] = df['Uptrend'] & (df['Low'] <= df['BB_Lower']) & (df['RSI'] < rsi_thresh)
            else:
                df['Tech_Valid'] = df['Uptrend'] & (df['RSI'] < rsi_thresh)
            df['Entry_Signal'] = df['Tech_Valid'] & df['Candle_Signal']

    # Patch position sizing ATR mult
    import strategy
    original_calc = strategy.calculate_position_size
    def custom_pos_size(equity, risk_pct, entry_price, atr, vix_multiplier):
        if atr <= 0 or vix_multiplier == 0: return 0, 0.0
        stop_loss = entry_price - (atr_mult * atr)
        if (entry_price - stop_loss) / entry_price > 0.10: return 0, stop_loss
        risk_per_share = entry_price - stop_loss
        shares = int(np.floor((equity * risk_pct / risk_per_share) * vix_multiplier))
        return shares, stop_loss
    strategy.calculate_position_size = custom_pos_size

    engine = BacktestEngine(data_dict, initial_capital=1000000.0, max_positions=5, risk_pct=0.02)
    engine.run()
    
    strategy.calculate_position_size = original_calc
    
    res = calculate_metrics(engine.equity_curve, engine.closed_trades)
    return {
        'rsi': rsi_thresh,
        'atr': atr_mult,
        'sector': use_sector,
        'bb': bb_touch,
        'cagr': res.get('CAGR (%)', 0),
        'trades': res.get('Total Trades', 0),
        'win_rate': res.get('Win Rate (%)', 0)
    }

params = {
    'rsi': [45, 50, 55, 60],
    'atr': [1.5, 2.0, 2.5],
    'sector': [True, False],
    'bb': [True, False]
}

keys, values = zip(*params.items())
experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"Running {len(experiments)} experiments...")
results = []
for p in experiments:
    try:
        r = run_sim(p['rsi'], p['atr'], p['sector'], p['bb'])
        results.append(r)
    except Exception as e:
        print(f"Error on {p}: {e}")

res_df = pd.DataFrame(results)
res_df = res_df.sort_values('cagr', ascending=False)
print("\nTop 5 Configs:")
print(res_df.head())

