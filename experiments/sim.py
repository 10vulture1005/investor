from strategy import Trade
from backtest_engine import BacktestEngine

import pandas as pd
date = pd.Timestamp('2020-01-01')
trade = Trade('TEST', date, 100, 92, 112, 120, 6250, 15, 'Auto')

# Simulate Win (T1 then T2)
print("WIN SIMULATION:")
cash = 1000000
trade_value = 6250 * 100
costs = trade_value * 0.0013
cash -= (trade_value + costs)
print(f"Bought. Cash = {cash}")

# Hit T1
prev_size = trade.current_size
pnl = trade.update(pd.Timestamp('2020-01-02'), 112, 112, 110, 112)
size_exited = prev_size - trade.current_size
exit_price = trade.exit_events[-1]['price']
value_sold = size_exited * exit_price
costs = value_sold * 0.0023
cash += value_sold - costs
print(f"T1 Hit. Pnl returned: {pnl}, size exited: {size_exited}, value sold: {value_sold}. Cash = {cash}")

# Hit T2
prev_size = trade.current_size
pnl = trade.update(pd.Timestamp('2020-01-03'), 120, 120, 115, 120)
size_exited = prev_size - trade.current_size
exit_price = trade.exit_events[-1]['price']
value_sold = size_exited * exit_price
costs = value_sold * 0.0023
cash += value_sold - costs
print(f"T2 Hit. Pnl returned: {pnl}, size exited: {size_exited}, value sold: {value_sold}. Cash = {cash}")

# Simulate Same Day T1 + T2
print("\nSAME DAY WIN SIMULATION:")
trade2 = Trade('TEST', date, 100, 92, 112, 120, 6250, 15, 'Auto')
cash = 1000000
trade_value = 6250 * 100
costs = trade_value * 0.0013
cash -= (trade_value + costs)
print(f"Bought. Cash = {cash}")

prev_size = trade2.current_size
pnl = trade2.update(pd.Timestamp('2020-01-02'), 120, 120, 110, 120)
size_exited = prev_size - trade2.current_size
exit_price = trade2.exit_events[-1]['price']
value_sold = size_exited * exit_price
costs = value_sold * 0.0023
cash += value_sold - costs
print(f"Same Day T1+T2 Hit. Pnl returned: {pnl}, size exited: {size_exited}, value sold: {value_sold}. exit_price: {exit_price}. Cash = {cash}")
