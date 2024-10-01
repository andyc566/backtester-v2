# data/alternative_data.py

import pandas as pd
import numpy as np

def generate_weather_data(start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date)
    temperatures = np.random.normal(60, 15, len(date_range))
    return pd.DataFrame({'Date': date_range, 'Temperature': temperatures}).set_index('Date')

def generate_storage_data(start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date)
    storage = np.cumsum(np.random.normal(0, 100, len(date_range))) + 3000  # Starting from 3000 Bcf
    return pd.DataFrame({'Date': date_range, 'Storage': storage}).set_index('Date')
