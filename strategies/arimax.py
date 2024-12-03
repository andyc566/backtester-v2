import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
import matplotlib.pyplot as plt

# Step 1: Load data
def load_data(file_path, date_col="Date", target_col="Price"):
    """
    Load time series data with exogenous variables from an Excel file.
    :param file_path: Path to the Excel file
    :param date_col: Name of the date column
    :param target_col: Name of the target (dependent) variable column
    :return: DataFrame with Date as index and target + exogenous variables
    """
    df = pd.read_excel(file_path)
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    df = df.sort_index()
    return df

# Step 2: Find optimal ARIMAX parameters using auto_arima
def find_optimal_arimax_params(df, target_col, exog_cols):
    """
    Find optimal ARIMAX parameters using auto_arima.
    :param df: DataFrame with time series and exogenous variables
    :param target_col: Target (dependent) variable
    :param exog_cols: List of exogenous variable column names
    :return: Optimal ARIMAX order and fitted model
    """
    exog = df[exog_cols] if exog_cols else None
    auto_model = auto_arima(
        df[target_col],
        exogenous=exog,
        start_p=0, max_p=5,
        start_q=0, max_q=5,
        d=None,
        seasonal=False,
        trace=True,
        stepwise=True,
        suppress_warnings=True,
        information_criterion="aic"
    )
    return auto_model.order, auto_model

# Step 3: Fit ARIMAX model
def fit_arimax(df, target_col, exog_cols, order):
    """
    Fit an ARIMAX model with given parameters.
    :param df: DataFrame with time series and exogenous variables
    :param target_col: Target (dependent) variable
    :param exog_cols: List of exogenous variable column names
    :param order: ARIMAX order (p, d, q)
    :return: Fitted ARIMAX model
    """
    exog = df[exog_cols] if exog_cols else None
    model = SARIMAX(df[target_col], exog=exog, order=order)
    fitted_model = model.fit(disp=False)
    return fitted_model

# Step 4: Generate trade signals
def generate_signals(df, forecast_col, target_col):
    """
    Generate trade signals based on forecast vs. actual price.
    :param df: DataFrame with actual and forecasted prices
    :param forecast_col: Name of the forecast column
    :param target_col: Name of the target (dependent) variable column
    :return: DataFrame with trade signals
    """
    df["Signal"] = np.where(df[forecast_col] > df[target_col], 1, -1)  # 1 for long, -1 for short
    return df

# Step 5: Backtest strategy
def backtest(df, target_col):
    """
    Backtest the strategy based on generated signals.
    :param df: DataFrame with actual and forecasted prices and signals
    :param target_col: Name of the target (dependent) variable column
    :return: DataFrame with cumulative returns and performance metrics
    """
    # Calculate daily returns
    df["Return"] = df[target_col].pct_change().fillna(0)
    
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

# Step 6: Visualization
def plot_results(df):
    """
    Visualize cumulative returns.
    :param df: DataFrame with cumulative returns
    """
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df["Cumulative Market Return"], label="Market Return", linestyle="--")
    plt.plot(df.index, df["Cumulative Strategy Return"], label="Strategy Return")
    plt.title("Cumulative Returns")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.show()

# Example usage
if __name__ == "__main__":
    # File path to the Excel file (update with your file path)
    file_path = "time_series_data_with_exog.xlsx"
    df = load_data(file_path)

    # Define target and exogenous variables
    target_col = "Price"
    exog_cols = [col for col in df.columns if col != target_col]

    # Find optimal ARIMAX parameters
    order, auto_model = find_optimal_arimax_params(df, target_col, exog_cols)
    print("Optimal ARIMAX order:", order)

    # Fit ARIMAX model
    arimax_model = fit_arimax(df, target_col, exog_cols, order)
    df["Forecast"] = arimax_model.predict(start=0, end=len(df) - 1, exog=df[exog_cols])

    # Generate trade signals
    df = generate_signals(df, "Forecast", target_col)

    # Backtest strategy
    df, metrics = backtest(df, target_col)

    # Display metrics
    print("\nBacktesting Performance Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")

    # Plot results
    plot_results(df)
