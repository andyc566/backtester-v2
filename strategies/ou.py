import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from typing import Tuple


class DataLoader:
    """Handles loading, merging, and preprocessing time series data."""
    def __init__(self, aeco_file: str, nymex_file: str):
        self.aeco_file = aeco_file
        self.nymex_file = nymex_file
        self.data = None

    def load_and_merge_data(self) -> pd.DataFrame:
        """Load AECO and NYMEX price data from Excel files and merge by date."""
        # Load data
        aeco_data = pd.read_excel(self.aeco_file)
        nymex_data = pd.read_excel(self.nymex_file)

        # Ensure date columns are in datetime format
        aeco_data['Date'] = pd.to_datetime(aeco_data['Date'])
        nymex_data['Date'] = pd.to_datetime(nymex_data['Date'])

        # Merge the data on the Date column
        merged_data = pd.merge(aeco_data, nymex_data, on='Date', suffixes=('_AECO', '_NYMEX'))

        # Drop rows where either AECO or NYMEX price is missing
        self.data = merged_data.dropna()
        return self.data

    def preprocess_data(self) -> pd.DataFrame:
        """Calculate the AECO-NYMEX spread."""
        self.data['spread'] = self.data['Price_AECO'] - self.data['Price_NYMEX']
        return self.data[['Date', 'spread']].set_index('Date')


class OrnsteinUhlenbeck:
    """Implements the Ornstein-Uhlenbeck process to model mean reversion."""
    def __init__(self, spread: pd.Series):
        self.spread = spread
        self.params = None

    def fit(self) -> Tuple[float, float, float]:
        """Estimate OU parameters (mean, speed of reversion, volatility)."""
        
        def ou_log_likelihood(params):
            mu, theta, sigma = params
            diff = self.spread.diff().dropna()
            xt = self.spread.shift(1).dropna()
            log_likelihood = (
                -0.5 * np.log(2 * np.pi * sigma**2) 
                - ((diff - theta * (mu - xt))**2 / (2 * sigma**2))
            ).sum()
            return -log_likelihood

        # Initial guesses
        mu0, theta0, sigma0 = self.spread.mean(), 0.1, self.spread.std()
        result = minimize(ou_log_likelihood, [mu0, theta0, sigma0], bounds=((None, None), (0, None), (0, None)))
        
        if result.success:
            self.params = result.x
            return self.params
        else:
            raise ValueError("OU Parameter estimation failed.")

    def generate_signals(self) -> pd.Series:
        """Generate mean reversion signals based on OU model."""
        if self.params is None:
            self.fit()
        mu, theta, sigma = self.params
        z_score = (self.spread - mu) / sigma
        return pd.Series(np.where(z_score > 1, -1, np.where(z_score < -1, 1, 0)), index=self.spread.index)


class TradingStrategy:
    """Manages trading signals and portfolio positions."""
    def __init__(self, signals: pd.Series, spread: pd.Series):
        self.signals = signals
        self.spread = spread
        self.returns = None

    def backtest(self) -> pd.DataFrame:
        """Backtest the strategy based on signals and calculate returns."""
        position = self.signals.shift(1).fillna(0)  # Enter on the next time step
        self.returns = position * self.spread.pct_change().fillna(0)
        return pd.DataFrame({"signals": self.signals, "returns": self.returns})

    def performance_metrics(self) -> dict:
        """Calculate performance metrics for the strategy."""
        cumulative_return = (1 + self.returns).prod() - 1
        annualized_return = (1 + cumulative_return) ** (252 / len(self.returns)) - 1
        annualized_vol = self.returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_vol if annualized_vol != 0 else np.nan
        return {
            "cumulative_return": cumulative_return,
            "annualized_return": annualized_return,
            "annualized_vol": annualized_vol,
            "sharpe_ratio": sharpe_ratio
        }


# Example Usage:
# Replace 'merged_aeco_prices.xlsx' and 'merged_nymex_prices.xlsx' with the paths to your files
aeco_file = 'merged_aeco_prices.xlsx'
nymex_file = 'merged_nymex_prices.xlsx'

# Load and preprocess data
data_loader = DataLoader(aeco_file, nymex_file)
merged_data = data_loader.load_and_merge_data()
spread_data = data_loader.preprocess_data()

# Fit OU model and generate signals
ou_model = OrnsteinUhlenbeck(spread_data['spread'])
signals = ou_model.generate_signals()

# Backtest the strategy
strategy = TradingStrategy(signals, spread_data['spread'])
backtest_results = strategy.backtest()
performance = strategy.performance_metrics()

# Output performance metrics
print(performance)
