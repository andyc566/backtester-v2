# data/data_loader.py

import pandas as pd

def load_spot_prices(file_path):
    df = pd.read_excel(file_path, sheet_name='Data 1', parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    return df

def load_futures_prices(file_path):
    df = pd.read_excel(file_path, sheet_name='Data 2', parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    return df