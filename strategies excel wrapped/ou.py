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
    """Implements the Ornstein-Uhlenbeck process to model mean reversion with take-profit criteria."""
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

    def generate_signals(self, z_entry_buy_threshold: float = -1.0, z_entry_sell_threshold: float = 1.0, z_exit_threshold: float = 0.5) -> pd.DataFrame:
        """Generate mean reversion signals with take-profit thresholds."""
        if self.params is None:
            self.fit()
        mu, theta, sigma = self.params
        z_score = (self.spread - mu) / sigma

        signals = np.zeros(len(z_score))
        positions = np.zeros(len(z_score))

        for i in range(1, len(z_score)):
            if positions[i - 1] == 0:  # No position
                if z_score[i] > z_entry_sell_threshold:  # Enter short
                    signals[i] = -1
                    positions[i] = -1
                elif z_score[i] < z_entry_buy_threshold:  # Enter long
                    signals[i] = 1
                    positions[i] = 1
            elif positions[i - 1] == 1:  # Long position
                if z_score[i] >= -z_exit_threshold:  # Close long
                    signals[i] = 0
                    positions[i] = 0
                else:
                    positions[i] = positions[i - 1]
            elif positions[i - 1] == -1:  # Short position
                if z_score[i] <= z_exit_threshold:  # Close short
                    signals[i] = 0
                    positions[i] = 0
                else:
                    positions[i] = positions[i - 1]

        return pd.DataFrame({'signals': signals, 'positions': positions}, index=self.spread.index)


class TradingStrategy:
    """Manages trading signals and portfolio positions."""
    def __init__(self, signals: pd.DataFrame, spread: pd.Series):
        self.signals = signals
        self.spread = spread
        self.returns = None

    def backtest(self) -> pd.DataFrame:
        """Backtest the strategy based on signals and calculate returns."""
        position = self.signals['positions'].shift(1).fillna(0)  # Enter on the next time step
        self.returns = position * self.spread.pct_change().fillna(0)
        return pd.DataFrame({"signals": self.signals['signals'], "positions": self.signals['positions'], "returns": self.returns})

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
        """Plot the spread and highlight buy, sell, and close signals."""
        buy_signals = self.signals[self.signals['signals'] == 1].index
        sell_signals = self.signals[self.signals['signals'] == -1].index
        close_signals = self.signals[
            (self.signals['signals'] == 0) & (self.signals['positions'].shift(1) != 0)
        ].index

        plt.figure(figsize=(14, 7))
        plt.plot(self.spread, label="Spread", color="blue")
        plt.scatter(buy_signals, self.spread[buy_signals], marker="^", color="green", label="Buy Signal")
        plt.scatter(sell_signals, self.spread[sell_signals], marker="v", color="red", label="Sell Signal")
        plt.scatter(close_signals, self.spread[close_signals], marker="o", color="orange", label="Close Position")
        plt.title("Spread with Buy, Sell, and Close Signals")
        plt.xlabel("Date")
        plt.ylabel("Spread")
        plt.legend()
        plt.grid()
        plt.show()


def optimize_thresholds(spread: pd.Series, ou_model: OrnsteinUhlenbeck) -> dict:
    """Optimize entry and exit z-score thresholds to maximize Sharpe ratio."""
    def objective(params):
        z_entry_buy, z_entry_sell, z_exit = params
        signals = ou_model.generate_signals(z_entry_buy, z_entry_sell, z_exit)
        strategy = TradingStrategy(signals, spread)
        backtest_results = strategy.backtest()
        metrics = strategy.performance_metrics()
        return -metrics["sharpe_ratio"]  # Minimize negative Sharpe ratio

    # Initial guesses and bounds
    initial_guess = [-1.0, 1.0, 0.5]
    bounds = [(-3.0, 0.0), (0.0, 3.0), (0.1, 1.0)]  # Buy < 0, Sell > 0, Exit > 0

    # Optimize
    result = minimize(objective, initial_guess, bounds=bounds)

    if result.success:
        optimal_thresholds = {
            "z_entry_buy_threshold": result.x[0],
            "z_entry_sell_threshold": result.x[1],
            "z_exit_threshold": result.x[2]
        }
        return optimal_thresholds
    else:
        raise ValueError("Optimization failed.")


# Example Usage
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
signals = ou_model.generate_signals(
    optimal_thresholds["z_entry_buy_threshold"],
    optimal_thresholds["z_entry_sell_threshold"],
    optimal_thresholds["z_exit_threshold"]
)

# Backtest the strategy
strategy = TradingStrategy(signals, spread_data['spread'])
backtest_results = strategy.backtest()
performance = strategy.performance_metrics()

# Output performance metrics
print(performance)

# Plot trading signals
strategy.plot_trading_signals()
