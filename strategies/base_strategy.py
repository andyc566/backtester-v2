# strategies/base_strategy.py

from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    def __init__(self, data):
        self.data = data
    
    @abstractmethod
    def generate_signals(self):
        pass