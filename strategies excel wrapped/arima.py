import os
import sys
import logging
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

# Setup logging
logging.basicConfig(
    filename='debug_log.txt',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

try:
    # Log start of the script
    logging.info("Starting ARIMA Python script.")

    # Validate command-line arguments
    if len(sys.argv) < 2:
        raise ValueError("Missing file path argument. Usage: python arima.py <file_path>")
    
    # Log file path
    file_path = os.path.abspath(sys.argv[1])
    logging.info(f"Received file path: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load data
    logging.info("Loading Excel data...")
    df = pd.read_excel(file_path)
    logging.info(f"Data loaded successfully. Data shape: {df.shape}")

    # Validate data
    if df.empty:
        raise ValueError("The loaded DataFrame is empty.")
    
    if "Date" not in df.columns or "Price" not in df.columns:
        raise ValueError("Expected columns 'Date' and 'Price' not found.")
    
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].isna().any():
        raise ValueError("Invalid dates detected in 'Date' column.")
    
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)

    # Check for missing values
    logging.info("Checking for missing values in the 'Price' column...")
    if df["Price"].isna().any():
        logging.warning("Missing values detected. Filling missing values...")
        df["Price"] = df["Price"].fillna(method="ffill").fillna(method="bfill")
        if df["Price"].isna().any():
            raise ValueError("Could not handle missing values in the 'Price' column.")
    logging.info("Missing values handled.")

    # Run ARIMA model
    logging.info("Running auto_arima to determine optimal ARIMA parameters...")
    auto_model = auto_arima(
        df["Price"],
        start_p=0, max_p=5,
        start_q=0, max_q=5,
        d=None,
        seasonal=False,
        trace=True,  # Include detailed ARIMA logs in stdout
        stepwise=True,
        suppress_warnings=True,
        information_criterion="aic"
    )
    order = auto_model.order
    logging.info(f"Optimal ARIMA order: {order}")

    arima_model = ARIMA(df["Price"], order=order).fit()
    logging.info("ARIMA model fitted successfully.")
    df["Forecast"] = arima_model.predict(start=0, end=len(df) - 1)
    logging.info("Forecast generated.")

    # Generate trading signals
    logging.info("Generating trading signals...")
    df["Signal"] = np.where(df["Forecast"] > df["Price"], 1, -1)
    logging.info("Trading signals generated.")

    # Backtest
    logging.info("Starting backtest...")
    df["Return"] = df["Price"].pct_change().fillna(0)
    df["Strategy Return"] = df["Signal"].shift(1) * df["Return"]
    df["Cumulative Market Return"] = (1 + df["Return"]).cumprod()
    df["Cumulative Strategy Return"] = (1 + df["Strategy Return"]).cumprod()

    # Calculate metrics
    logging.info("Calculating backtest metrics...")
    strategy_volatility = df["Strategy Return"].std() * np.sqrt(252)
    sharpe_ratio = df["Strategy Return"].mean() / df["Strategy Return"].std() * np.sqrt(252)
    logging.info(f"Backtest metrics calculated. Volatility: {strategy_volatility:.2f}, Sharpe Ratio: {sharpe_ratio:.2f}")

    # Write metrics to output file
    output_file = os.path.join(os.path.dirname(__file__), "metrics_output.txt")
    logging.info(f"Writing metrics to {output_file}...")
    with open(output_file, "w") as f:
        f.write(f"Volatility (Strategy): {strategy_volatility:.2f}\n")
        f.write(f"Sharpe Ratio: {sharpe_ratio:.2f}\n")
    logging.info("Metrics successfully written to file.")

except Exception as e:
    # Log critical errors and exit
    logging.critical(f"Critical Error: {e}", exc_info=True)
    sys.exit(1)

# Log end of script
logging.info("ARIMA script completed successfully.")
