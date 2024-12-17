import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from openpyxl import Workbook
from openpyxl.styles import PatternFill
from pulp import *
import os

class StorageConfig:
    def __init__(self,
                 max_capacity=1000000,
                 injection_thresholds=[0, 500000, 800000, 1000000],
                 injection_rates=[12500, 7500, 5000],
                 withdrawal_thresholds=[0, 200000, 500000, 1000000],
                 withdrawal_rates=[5000, 7500, 12500],
                 winter_months=[10, 11, 12, 1, 2, 3],
                 prices=None):
        self.max_capacity = max_capacity
        self.injection_thresholds = injection_thresholds
        self.injection_rates = injection_rates
        self.withdrawal_thresholds = withdrawal_thresholds
        self.withdrawal_rates = withdrawal_rates
        self.winter_months = winter_months
        self.summer_months = [m for m in range(1, 13) if m not in winter_months]
        self.prices = prices or {}

def read_risk_free_rates(file_path):
    if os.path.exists(file_path):
        df = pd.read_excel(file_path, sheet_name='RiskFreeRates')
        return df.set_index('Month')['Rate'].to_dict()
    else:
        print(f"Warning: {file_path} not found. Using default risk-free rates.")
        default_rates = {
            '2025-04': 2.0, '2025-05': 2.0, '2025-06': 2.0,
            '2025-07': 2.0, '2025-08': 2.0, '2025-09': 2.0,
            '2025-10': 2.0, '2025-11': 2.0, '2025-12': 2.0,
            '2026-01': 2.0, '2026-02': 2.0, '2026-03': 2.0
        }
        return default_rates

def read_prices(file_path):
    if os.path.exists(file_path):
        df = pd.read_excel(file_path, sheet_name='Prices')
        return df.set_index('Month')['Price'].to_dict()
    else:
        print(f"Warning: {file_path} not found. Using default prices.")
        default_prices = {
            '2025-04': 1.603, '2025-05': 1.507, '2025-06': 1.588,
            '2025-07': 1.673, '2025-08': 1.752, '2025-09': 1.761,
            '2025-10': 2.030, '2025-11': 2.707, '2025-12': 3.121,
            '2026-01': 3.332, '2026-02': 3.223, '2026-03': 2.771
        }
        return default_prices

def create_price_schedule(config):
    start_date = datetime(2025, 4, 1)
    dates = []
    daily_prices = []
    
    for _ in range(365):
        month_key = start_date.strftime('%Y-%m')
        dates.append(start_date)
        daily_prices.append(config.prices.get(month_key, 0))
        start_date += timedelta(days=1)
    
    return pd.DataFrame({'Date': dates, 'Price': daily_prices})

def apply_daily_discount(df, risk_free_rates):
    df['MonthKey'] = df['Date'].dt.strftime('%Y-%m')
    df['DaysSinceStart'] = (df['Date'] - df['Date'].min()).dt.days
    df['DiscountFactor'] = df.apply(
        lambda row: (1 + risk_free_rates.get(row['MonthKey'], 0) / 100) ** (-row['DaysSinceStart'] / 365), axis=1
    )
    df['Discounted_Price'] = df['Price'] * df['DiscountFactor']
    return df

def optimize_storage(config, risk_free_rates):
    df = create_price_schedule(config)
    df = apply_daily_discount(df, risk_free_rates)
    df['Month'] = df['Date'].dt.month
    df['Actual Injection'] = 0.0
    df['Actual Withdrawal'] = 0.0

    prob = LpProblem("Natural_Gas_Storage_Optimization", LpMaximize)
    days = range(len(df))

    injections = LpVariable.dicts("injection", days, lowBound=0, cat='Continuous')
    withdrawals = LpVariable.dicts("withdrawal", days, lowBound=0, cat='Continuous')
    storage = LpVariable.dicts("storage", days, lowBound=0, upBound=config.max_capacity, cat='Continuous')

    prob += lpSum([withdrawals[i] * df['Discounted_Price'].iloc[i] - injections[i] * df['Discounted_Price'].iloc[i] for i in days])

    M = config.max_capacity

    for i in days:
        month = df['Month'].iloc[i]
        if i == 0:
            prob += storage[i] == df['Actual Injection'].iloc[i] - df['Actual Withdrawal'].iloc[i]
        else:
            prob += storage[i] == storage[i-1] + injections[i] - withdrawals[i]
        
        # Ensure correct injection rates based on storage
        for j in range(len(config.injection_thresholds) - 1):
            prob += injections[i] <= config.injection_rates[j] * lpSum(
                [1 for k in range(len(config.injection_thresholds) - 1) 
                 if storage[i-1] >= config.injection_thresholds[k] and storage[i-1] < config.injection_thresholds[k+1] and k == j])
        
        # Ensure correct withdrawal rates based on storage
        for j in range(len(config.withdrawal_thresholds) - 1):
            prob += withdrawals[i] <= config.withdrawal_rates[j] * lpSum(
                [1 for k in range(len(config.withdrawal_thresholds) - 1) 
                 if storage[i-1] >= config.withdrawal_thresholds[k] and storage[i-1] < config.withdrawal_thresholds[k+1] and k == j])

    prob.solve()

    df['Optimized Storage'] = [storage[i].value() for i in days]
    df['Optimized Injection'] = [injections[i].value() for i in days]
    df['Optimized Withdrawal'] = [withdrawals[i].value() for i in days]
    df['Daily_Profit'] = df['Optimized Withdrawal'] * df['Discounted_Price'] - df['Optimized Injection'] * df['Discounted_Price']
    return df

def export_to_excel(df, filename='natgas_storage_optimization.xlsx'):
    writer = pd.ExcelWriter(filename, engine='openpyxl')
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    df.to_excel(writer, index=False, sheet_name='Storage_Schedule')

    workbook = writer.book
    worksheet = writer.sheets['Storage_Schedule']

    header_fill = PatternFill(start_color='CCCCCC', end_color='CCCCCC', fill_type='solid')

    for cell in worksheet[1]:
        cell.fill = header_fill

    for column in worksheet.columns:
        max_length = 0
        column = list(column)
        for cell in column:
            if cell.value and len(str(cell.value)) > max_length:
                max_length = len(str(cell.value))
        adjusted_width = (max_length + 2)
        worksheet.column_dimensions[column[0].column_letter].width = adjusted_width

    writer.close()

prices = read_prices('prices.xlsx')
custom_config = StorageConfig(prices=prices)

risk_free_rates = read_risk_free_rates('risk_free_rates.xlsx')
df_result = optimize_storage(custom_config, risk_free_rates)
export_to_excel(df_result)

total_profit = df_result['Daily_Profit'].sum()
print(f"Total Profit: ${total_profit:,.2f}")
print("Results exported to natgas_storage_optimization.xlsx")
