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
nifty_df['Regime_Allowed'] = calculate_market_regime_filter(nifty_df)

def add_techs(df):
    df = df.copy()
    df['Prev_Close'] = df['Close'].shift(1)
    df['TR'] = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Prev_Close']), abs(df['Low'] - df['Prev_Close'])))
    df['ATR'] = df['TR'].rolling(window=14).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['Uptrend'] = df['Close'] > df['EMA_50']
    
    df['Pullback'] = (df['Low'] <= df['EMA_20']) & (df['Close'] > df['EMA_20'])
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

def run_sim(risk_pct, use_sector, trail_20ema, tighten_regime):
    os.environ['RISK_PCT'] = str(risk_pct)
    os.environ['MAX_POSITIONS'] = '5'
    
    if tighten_regime:
        nifty_df['Regime_Allowed'] = nifty_df['Close'] > nifty_df['Close'].ewm(span=20, adjust=False).mean()
    else:
        nifty_df['Regime_Allowed'] = calculate_market_regime_filter(nifty_df)

    from data_fetcher import SECTOR_INDICES
    if use_sector:
        from filters import calculate_sector_rs_filter
        for name in SECTOR_INDICES.keys():
            if name in data_dict:
                data_dict[name]['Sector_RS'] = calculate_sector_rs_filter(data_dict[name], nifty_df)
    else:
        for name in SECTOR_INDICES.keys():
            if name in data_dict:
                data_dict[name]['Sector_RS'] = True

    for ticker in data_dict:
        if ticker.endswith('.NS') and ticker != 'NIFTYBEES.NS':
            df = data_dict[ticker]
            df['Tech_Valid'] = df['Uptrend'] & df['Pullback']
            df['Entry_Signal'] = df['Tech_Valid'] & df['Candle_Signal']

    def custom_pos_size(equity, r_pct, entry_price, atr, vix_multiplier):
        if pd.isna(atr) or atr <= 0 or vix_multiplier == 0: return 0, 0.0
        stop_loss = entry_price - (1.5 * atr)
        if (entry_price - stop_loss) / entry_price > 0.15: return 0, stop_loss
        risk_per_share = entry_price - stop_loss
        if risk_per_share <= 0: return 0, 0.0
        shares = int(np.floor((equity * risk_pct / risk_per_share) * vix_multiplier))
        return shares, stop_loss
    strategy.calculate_position_size = custom_pos_size

    original_trade_update = strategy.Trade.update
    def custom_trade_update(self, date, open_p, high_p, low_p, close_p):
        realized_pnl = 0.0
        if self.status == 'CLOSED': return 0.0
        
        # dynamic stop: close below EMA 20
        if trail_20ema and self.hold_days > 2:
            df = data_dict[self.stock]
            if date in df.index:
                ema_20 = df.loc[date, 'EMA_20']
                if close_p < ema_20:
                    price = close_p
                    size_to_sell = self.current_size
                    pnl = (price - self.entry_price) * size_to_sell
                    realized_pnl += pnl
                    self.exit_events.append({'date': date, 'price': price, 'size': size_to_sell, 'reason': 'TRAIL_EMA', 'pnl': pnl})
                    self.current_size = 0
                    self.status = 'CLOSED'
                    return realized_pnl
                    
        return original_trade_update(self, date, open_p, high_p, low_p, close_p)
        
    strategy.Trade.update = custom_trade_update

    engine = BacktestEngine(data_dict, max_positions=5, risk_pct=risk_pct)
    if not use_sector:
        original_run = engine.run
        for name in SECTOR_INDICES.keys():
            if name in engine.data_dict:
                engine.data_dict[name]['Sector_RS'] = True

    engine.run()
    strategy.calculate_position_size = original_calc
    strategy.Trade.update = original_trade_update
    
    res = calculate_metrics(engine.equity_curve, engine.closed_trades)
    return {
        'risk_pct': risk_pct,
        'sector': use_sector,
        'trail_20': trail_20ema,
        'regime_20': tighten_regime,
        'cagr': float(str(res.get('CAGR (%)', '0')).replace('%', '')),
        'trades': res.get('Total Trades', 0),
        'win_rate': res.get('Win Rate (%)', '0%'),
        'dd': float(str(res.get('Max Drawdown (%)', '0')).replace('%', ''))
    }

params = {
    'risk_pct': [0.02, 0.03],
    'use_sector': [True, False],
    'trail_20ema': [True, False],
    'tighten_regime': [True, False]
}
keys, values = zip(*params.items())
experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"Running {len(experiments)} experiments...")
results = []
for p in experiments:
    r = run_sim(**p)
    results.append(r)

res_df = pd.DataFrame(results)
res_df = res_df.sort_values('dd', ascending=False) # Highest (closest to 0) drawdown
print("\nTop Configs by Drawdown:")
print(res_df.head(10).to_string())

