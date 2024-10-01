# strategies/example_strategies.py

from .base_strategy import BaseStrategy
import pandas as pd
import numpy as np 

class MovingAverageCrossover(BaseStrategy):
    def __init__(self, data, short_window=10, long_window=30):
        super().__init__(data)
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signals(self):
        signals = pd.DataFrame(index=self.data.index)
        signals['signal'] = 0.0
        
        signals['short_mavg'] = self.data['Henry Hub Natural Gas Spot Price (Dollars per Million Btu)'].rolling(window=self.short_window, min_periods=1, center=False).mean()
        signals['long_mavg'] = self.data['Henry Hub Natural Gas Spot Price (Dollars per Million Btu)'].rolling(window=self.long_window, min_periods=1, center=False).mean()
        
        signals['signal'][self.short_window:] = np.where(signals['short_mavg'][self.short_window:] > signals['long_mavg'][self.short_window:], 1.0, 0.0)
        signals['positions'] = signals['signal'].diff()
        
        return signals