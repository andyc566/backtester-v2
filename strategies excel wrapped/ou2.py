import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
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

    def generate_signals(self, z_buy_threshold: float = -1.0, z_sell_threshold: float = 1.0) -> pd.Series:
        """Generate mean reversion signals based on OU model with custom thresholds."""
        if self.params is None:
            self.fit()
        mu, theta, sigma = self.params
        z_score = (self.spread - mu) / sigma

        # Generate signals based on thresholds
        signals = np.where(z_score > z_sell_threshold, -1, np.where(z_score < z_buy_threshold, 1, 0))
        return pd.Series(signals, index=self.spread.index)


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

    def plot_trading_signals(self):
        """Plot the spread and highlight buy and sell signals."""
        buy_signals = self.signals[self.signals == 1].index
        sell_signals = self.signals[self.signals == -1].index

        plt.figure(figsize=(14, 7))
        plt.plot(self.spread, label="Spread", color="blue")
        plt.scatter(buy_signals, self.spread[buy_signals], marker="^", color="green", label="Buy Signal")
        plt.scatter(sell_signals, self.spread[sell_signals], marker="v", color="red", label="Sell Signal")
        plt.title("Spread with Buy and Sell Signals")
        plt.xlabel("Date")
        plt.ylabel("Spread")
        plt.legend()
        plt.grid()
        plt.show()


def optimize_thresholds(spread: pd.Series, ou_model: OrnsteinUhlenbeck) -> dict:
    """Optimize z-score thresholds for buying and selling to maximize Sharpe ratio."""
    def objective(params):
        z_buy_threshold, z_sell_threshold = params
        signals = ou_model.generate_signals(z_buy_threshold, z_sell_threshold)
        strategy = TradingStrategy(signals, spread)
        backtest_results = strategy.backtest()
        metrics = strategy.performance_metrics()
        return -metrics["sharpe_ratio"]  # Minimize the negative Sharpe ratio

    # Initial guesses and bounds for thresholds
    initial_guess = [-1.0, 1.0]
    bounds = [(-3.0, 0.0), (0.0, 3.0)]  # Buy threshold < 0, Sell threshold > 0

    # Optimize
    result = minimize(objective, initial_guess, bounds=bounds)

    if result.success:
        optimal_thresholds = {"z_buy_threshold": result.x[0], "z_sell_threshold": result.x[1]}
        return optimal_thresholds
    else:
        raise ValueError("Optimization failed.")


# Example Usage:
aeco_file = 'merged_aeco_prices.xlsx'
nymex_file = 'merged_nymex_prices.xlsx'

# Load and preprocess data
data_loader = DataLoader(aeco_file, nymex_file)
merged_data = data_loader.load_and_merge_data()
spread_data = data_loader.preprocess_data()

# Fit OU model
ou_model = OrnsteinUhlenbeck(spread_data['spread'])

# Optimize z-score thresholds
optimal_thresholds = optimize_thresholds(spread_data['spread'], ou_model)
print("Optimal Thresholds:", optimal_thresholds)

# Generate signals using optimal thresholds
signals = ou_model.generate_signals(optimal_thresholds["z_buy_threshold"], optimal_thresholds["z_sell_threshold"])

# Backtest the strategy
strategy = TradingStrategy(signals, spread_data['spread'])
backtest_results = strategy.backtest()
performance = strategy.performance_metrics()

# Output performance metrics
print(performance)

# Plot trading signals
strategy.plot_trading_signals()
