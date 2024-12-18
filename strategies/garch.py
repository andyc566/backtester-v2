import pandas as pd
import numpy as np
from arch import arch_model

def forecast_volatility(file_path):
    # Load data from Excel file
    df = pd.read_excel(file_path)
    
    # Ensure correct columns exist
    if 'Date' not in df.columns or 'Price' not in df.columns:
        raise ValueError("The Excel file must contain 'Date' and 'Price' columns.")

    # Calculate log returns
    df['Return'] = np.log(df['Price'] / df['Price'].shift(1)).dropna()

    # Define and fit the GARCH model
    model1 = arch_model(df['Return'].dropna(), vol='Garch', p=1, q=1, dist='Normal')
    results1 = model1.fit(disp='off')

    # Define and fit the EGARCH model
    model2 = arch_model(df['Return'].dropna(), vol='EGARCH', p=1, q=1, dist='Normal')
    results2 = model2.fit(disp='off')

    # Forecast future volatility
    forecasts1 = results1.forecast(horizon=1)
    forecast_variance1 = forecasts1.variance.iloc[-1]
    forecast_std_dev1 = np.sqrt(forecast_variance1)

    forecasts2 = results2.forecast(horizon=1)
    forecast_variance2 = forecasts2.variance.iloc[-1]
    forecast_std_dev2 = np.sqrt(forecast_variance2)

    # Print results
    print("GARCH Model Summary:")
    print(results1.summary())
    print("\nEGARCH Model Summary:")
    print(results2.summary())

    print("\n1-day Forecasted Volatility (Standard Deviation):")
    print("GARCH:", forecast_std_dev1)
    print("EGARCH:", forecast_std_dev2)

if __name__ == "__main__": 
    forecast_volatility('Henry_Hub_Natural_Gas_Spot_Price.xlsx')


'''
Distributions (dist parameter):
'Normal' (Gaussian Distribution): Assumes normally distributed returns. This is the default but may underestimate tail risk.
't' (Student's t-Distribution): Allows for heavier tails, making the model more robust to extreme events.
'Skewt' (Skewed Student's t-Distribution): Extends Student’s t-distribution to account for skewness as well as heavy tails.
'GED' (Generalized Error Distribution): Allows for flexible tail behavior, useful when returns exhibit excess kurtosis.
'''