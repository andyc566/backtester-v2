import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from openpyxl import Workbook
from openpyxl.styles import PatternFill

class StorageConfig:
    def __init__(self,
                 max_capacity=1000000,
                 low_threshold=500000,
                 high_threshold=750000,
                 injection_limit_low=7500,
                 injection_limit_mid=5000,
                 injection_limit_high=2500,
                 withdrawal_limit_low=2500,
                 withdrawal_limit_mid=5000,
                 withdrawal_limit_high=7500,
                 winter_months=[10, 11, 12, 1, 2, 3],
                 prices=None):
        
        self.max_capacity = max_capacity
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.injection_limit_low = injection_limit_low
        self.injection_limit_mid = injection_limit_mid
        self.injection_limit_high = injection_limit_high
        self.withdrawal_limit_low = withdrawal_limit_low
        self.withdrawal_limit_mid = withdrawal_limit_mid
        self.withdrawal_limit_high = withdrawal_limit_high
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

def get_injection_limit(current_storage, config):
    if current_storage < config.low_threshold:
        return config.injection_limit_low
    elif current_storage < config.high_threshold:
        return config.injection_limit_mid
    else:
        return config.injection_limit_high

def get_withdrawal_limit(current_storage, config):
    if current_storage > config.high_threshold:
        return config.withdrawal_limit_high
    elif current_storage > config.low_threshold:
        return config.withdrawal_limit_mid
    else:
        return config.withdrawal_limit_low

def optimize_storage(config):
    # Initialize data
    df = create_price_schedule(config)
    
    # Initialize columns with explicit dtypes
    df['Storage'] = 0
    df['Action'] = 0
    df['Daily_Profit'] = 0.0  # Explicitly set as float
    df['Season'] = 'Summer'
    
    # Define winter/summer seasons
    df.loc[df['Date'].dt.month.isin(config.winter_months), 'Season'] = 'Winter'
    
    current_storage = 0
    
    # Summer injection
    summer_mask = df['Season'] == 'Summer'
    for idx in df[summer_mask].index:
        injection_limit = get_injection_limit(current_storage, config)
        
        if current_storage < config.max_capacity:
            injection = min(injection_limit, config.max_capacity - current_storage)
            df.loc[idx, 'Action'] = injection
            current_storage += injection
            df.loc[idx, 'Storage'] = current_storage
            df.loc[idx, 'Daily_Profit'] = float(-injection * df.loc[idx, 'Price'])  # Explicit float conversion
    
    # Winter withdrawal
    winter_mask = df['Season'] == 'Winter'
    for idx in df[winter_mask].index:
        withdrawal_limit = get_withdrawal_limit(current_storage, config)
        
        if current_storage > 0:
            withdrawal = min(withdrawal_limit, current_storage)
            df.loc[idx, 'Action'] = -withdrawal
            current_storage -= withdrawal
            df.loc[idx, 'Storage'] = current_storage
            df.loc[idx, 'Daily_Profit'] = float(withdrawal * df.loc[idx, 'Price'])  # Explicit float conversion
    
    return df

def export_to_excel(df, filename='natgas_storage_optimization.xlsx'): # DEFINE YOUR PATH HERE
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

# Example usage with custom configuration
custom_config = StorageConfig(
    max_capacity=1000000,  # Increased capacity
    low_threshold=500000,  # Modified thresholds
    high_threshold=750000,
    injection_limit_low=10000,  # Modified injection limits
    injection_limit_mid=7500,
    injection_limit_high=5000,
    withdrawal_limit_low=5000,  # Modified withdrawal limits
    withdrawal_limit_mid=7500,
    withdrawal_limit_high=10000,
    winter_months=[10, 11, 12, 1, 2, 3],  # Modified winter season
    prices={  # Custom prices
        '2025-04': 2.2, '2025-05': 2.0, '2025-06': 2.1,
        '2025-07': 2.2, '2025-08': 2.3, '2025-09': 2.4,
        '2025-10': 3.7, '2025-11': 3.8, '2025-12': 3.9,
        '2026-01': 4.0, '2026-02': 4.1, '2026-03': 4.0
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