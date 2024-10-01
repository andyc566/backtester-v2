# backtester/backtester.py

import pandas as pd
import numpy as np

class Backtester:
    def __init__(self, data, strategy, initial_capital=100000):
        self.data = data
        self.strategy = strategy
        self.initial_capital = initial_capital
        
    def run_backtest(self):
        signals = self.strategy.generate_signals()
        portfolio = self.generate_portfolio(signals)
        return portfolio
    
    def generate_portfolio(self, signals):
        positions = pd.DataFrame(index=signals.index).fillna(0.0)
        portfolio = pd.DataFrame(index=signals.index).fillna(0.0)
        
        positions['NG'] = 100 * signals['signal']
        portfolio['positions'] = (positions.multiply(self.data['Henry Hub Natural Gas Spot Price (Dollars per Million Btu)'], axis=0))
        portfolio['cash'] = self.initial_capital - (positions.diff().multiply(self.data['Henry Hub Natural Gas Spot Price (Dollars per Million Btu)'], axis=0)).cumsum()
        portfolio['total'] = portfolio['positions'] + portfolio['cash']
        portfolio['returns'] = portfolio['total'].pct_change()
        
        return portfolio