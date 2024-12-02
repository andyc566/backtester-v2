import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

# Step 1: Load the time series data from Excel
def load_data(file_path):
    """
    Load time series data from an Excel file.
    :param file_path: Path to the Excel file
    :return: A DataFrame with Date as index and Price as the time series
    """
    df = pd.read_excel(file_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df = df.sort_index()
    return df

# File path to the Excel file (update with your file path)
file_path = "time_series_data.xlsx"
df = load_data(file_path)

# Step 2: Find optimal ARIMA parameters using auto_arima
auto_model = auto_arima(
    df["Price"],
    start_p=0, max_p=5,
    start_q=0, max_q=5,
    d=None,  # Let auto_arima determine d
    seasonal=False,  # Non-seasonal ARIMA
    trace=True,
    stepwise=True,
    suppress_warnings=True,
    information_criterion="aic"
)
print("Optimal ARIMA order:", auto_model.order)

# Step 3: Fit ARIMA model with optimal parameters
order = auto_model.order
arima_model = ARIMA(df["Price"], order=order).fit()
df["Forecast"] = arima_model.predict(start=0, end=len(df) - 1)

# Step 4: Generate trade signals based on model's forecast
df["Signal"] = np.where(df["Forecast"] > df["Price"], 1, -1)  # 1 for long, -1 for short

# Step 5: Backtesting function
def backtest(df):
    """
    Backtest the strategy based on generated signals.
    """
    # Calculate daily returns
    df["Return"] = df["Price"].pct_change().fillna(0)
    
    # Strategy returns based on signal
    df["Strategy Return"] = df["Signal"].shift(1) * df["Return"]
    
    # Cumulative returns
    df["Cumulative Market Return"] = (1 + df["Return"]).cumprod()
    df["Cumulative Strategy Return"] = (1 + df["Strategy Return"]).cumprod()
    
    # Metrics
    strategy_volatility = df["Strategy Return"].std() * np.sqrt(252)
    sharpe_ratio = df["Strategy Return"].mean() / df["Strategy Return"].std() * np.sqrt(252)
    metrics = {
        "Volatility (Strategy)": strategy_volatility,
        "Sharpe Ratio": sharpe_ratio
    }
    return df, metrics

# Apply backtesting
df, metrics = backtest(df)

# Step 6: Visualization
plt.figure(figsize=(14, 7))
plt.plot(df.index, df["Cumulative Market Return"], label="Market Return", linestyle="--")
plt.plot(df.index, df["Cumulative Strategy Return"], label="Strategy Return")
plt.title("Cumulative Returns")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.show()

# Display metrics
print("\nBacktesting Performance Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.2f}")
