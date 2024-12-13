import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from openpyxl import Workbook
from openpyxl.styles import PatternFill
from pulp import *

class StorageConfig:
    def __init__(self,
                 max_capacity=1000000,
                 # Injection thresholds and rates
                 injection_thresholds=[0, 500000, 800000, 1000000],
                 injection_rates=[12500, 7500, 5000],
                 # Withdrawal thresholds and rates
                 withdrawal_thresholds=[0, 200000, 500000, 1000000],
                 withdrawal_rates=[5000, 7500, 12500],
                 winter_months=[10, 11, 12, 1, 2, 3],
                 prices=None):
        
        self.max_capacity = max_capacity
        
        # Injection parameters
        self.injection_thresholds = injection_thresholds
        self.injection_rates = injection_rates
        
        # Withdrawal parameters
        self.withdrawal_thresholds = withdrawal_thresholds
        self.withdrawal_rates = withdrawal_rates
        
        self.winter_months = winter_months
        self.summer_months = [m for m in range(1, 13) if m not in winter_months]
        
        self.prices = prices or {
            '2025-04': 2.0, '2025-05': 1.8, '2025-06': 1.9,
            '2025-07': 2.0, '2025-08': 2.1, '2025-09': 2.2,
            '2025-10': 3.5, '2025-11': 3.6, '2025-12': 3.7,
            '2026-01': 3.8, '2026-02': 3.9, '2026-03': 3.8
        }

def create_price_schedule(config):
    start_date = datetime(2025, 4, 1)
    dates = []
    daily_prices = []
    
    for _ in range(365):
        month_key = start_date.strftime('%Y-%m')
        dates.append(start_date)
        daily_prices.append(config.prices[month_key])
        start_date += timedelta(days=1)
    
    return pd.DataFrame({'Date': dates, 'Price': daily_prices})

def optimize_storage(config):
    # Create price schedule
    df = create_price_schedule(config)
    df['Month'] = df['Date'].dt.month
    
    # Initialize optimization problem
    prob = LpProblem("Natural_Gas_Storage_Optimization", LpMaximize)
    
    # Create decision variables
    days = range(len(df))
    
    # Main variables
    injections = LpVariable.dicts("injection", days, lowBound=0, cat='Continuous')
    withdrawals = LpVariable.dicts("withdrawal", days, lowBound=0, cat='Continuous')
    storage = LpVariable.dicts("storage", days, lowBound=0, upBound=config.max_capacity, cat='Continuous')
    
    # Binary variables for storage levels
    # For injection thresholds
    injection_zone = LpVariable.dicts("injection_zone", 
                                    ((i, j) for i in days for j in range(len(config.injection_rates))),
                                    cat='Binary')
    
    # For withdrawal thresholds
    withdrawal_zone = LpVariable.dicts("withdrawal_zone", 
                                     ((i, j) for i in days for j in range(len(config.withdrawal_rates))),
                                     cat='Binary')
    
    # Big M constant
    M = config.max_capacity
    
    # Objective function: Maximize profit
    prob += lpSum([withdrawals[i] * df['Price'].iloc[i] - injections[i] * df['Price'].iloc[i] for i in days])
    
    # Constraints
    for i in days:
        month = df['Month'].iloc[i]
        
        # Storage balance constraints
        if i == 0:
            prob += storage[i] == injections[i] - withdrawals[i]
        else:
            prob += storage[i] == storage[i-1] + injections[i] - withdrawals[i]
        
        # Only one zone can be active at a time for each operation
        prob += lpSum(injection_zone[i,j] for j in range(len(config.injection_rates))) == 1
        prob += lpSum(withdrawal_zone[i,j] for j in range(len(config.withdrawal_rates))) == 1
        
        # Zone constraints for injection
        for j in range(len(config.injection_rates)):
            prob += storage[i] >= config.injection_thresholds[j] - M * (1 - injection_zone[i,j])
            prob += storage[i] <= config.injection_thresholds[j+1] + M * (1 - injection_zone[i,j])
        
        # Zone constraints for withdrawal
        for j in range(len(config.withdrawal_rates)):
            prob += storage[i] >= config.withdrawal_thresholds[j] - M * (1 - withdrawal_zone[i,j])
            prob += storage[i] <= config.withdrawal_thresholds[j+1] + M * (1 - withdrawal_zone[i,j])
        
        # Seasonal and rate constraints
        if month in config.winter_months:
            # Winter: only withdrawals allowed
            prob += injections[i] == 0
            # Set withdrawal limits based on storage zones
            prob += withdrawals[i] <= lpSum(config.withdrawal_rates[j] * withdrawal_zone[i,j] 
                                          for j in range(len(config.withdrawal_rates)))
        else:
            # Summer: only injections allowed
            prob += withdrawals[i] == 0
            # Set injection limits based on storage zones
            prob += injections[i] <= lpSum(config.injection_rates[j] * injection_zone[i,j] 
                                         for j in range(len(config.injection_rates)))
    
    # Solve the optimization problem
    prob.solve()
    
    if LpStatus[prob.status] != 'Optimal':
        print(f"Warning: Solution status is {LpStatus[prob.status]}")
    
    # Extract results
    df['Storage'] = [storage[i].value() for i in days]
    df['Injection'] = [injections[i].value() for i in days]
    df['Withdrawal'] = [withdrawals[i].value() for i in days]
    df['Action'] = df['Injection'] - df['Withdrawal']
    df['Daily_Profit'] = df['Withdrawal'] * df['Price'] - df['Injection'] * df['Price']
    df['Season'] = 'Summer'
    df.loc[df['Month'].isin(config.winter_months), 'Season'] = 'Winter'
    
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
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = (max_length + 2)
        worksheet.column_dimensions[column[0].column_letter].width = adjusted_width
    
    writer.close()

# Example usage with the specified thresholds and rates
custom_config = StorageConfig(
    max_capacity=1000000,
    # Injection parameters
    injection_thresholds=[0, 500000, 800000, 1000000],
    injection_rates=[12500, 7500, 5000],
    # Withdrawal parameters
    withdrawal_thresholds=[0, 200000, 500000, 1000000],
    withdrawal_rates=[5000, 7500, 12500],
    winter_months=[11, 12, 1, 2, 3],
    prices={
        '2025-04': 1.603, '2025-05': 1.507, '2025-06': 1.588,
        '2025-07': 1.673, '2025-08': 1.752, '2025-09': 1.761,
        '2025-10': 2.030, '2025-11': 2.707, '2025-12': 3.121,
        '2026-01': 3.332, '2026-02': 3.223, '2026-03': 2.771
    }
)

# Run the optimization with custom config
df_result = optimize_storage(custom_config)

# Calculate total profit
total_profit = df_result['Daily_Profit'].sum()

# Export results
export_to_excel(df_result)

print(f"Total Profit: ${total_profit:,.2f}")
print(f"Results exported to natgas_storage_optimization.xlsx")

