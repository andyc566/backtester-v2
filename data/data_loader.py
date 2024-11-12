import pandas as pd
from datetime import datetime

def process_excel_dates(file_path):
    """
    Process Excel file to match dates and extract corresponding prices.
    
    Args:
        file_path (str): Path to the Excel file
        
    Returns:
        pandas.DataFrame: DataFrame with matched dates and prices
    """
    # Read the Excel file, skipping the first two rows for the data
    df = pd.read_excel(file_path, index_col=0, skiprows=2)
    
    # Convert the index (first column dates) to datetime
    df.index = pd.to_datetime(df.index)
    
    # Convert column headers (dates) to datetime
    # First, get the header row
    headers = pd.read_excel(file_path, nrows=1)
    # Convert header dates starting from second column
    date_headers = pd.to_datetime(headers.columns[1:])
    
    # Rename columns with datetime objects
    df.columns = date_headers
    
    # Initialize lists to store matched dates and prices
    matched_dates = []
    matched_prices = []
    
    # Iterate through each row
    for row_date, row in df.iterrows():
        # Iterate through each column
        for col_date in df.columns:
            # Check if dates match
            if row_date == col_date:
                matched_dates.append(row_date)
                matched_prices.append(row[col_date])
    
    # Create result DataFrame
    result_df = pd.DataFrame({
        'Date': matched_dates,
        'Price': matched_prices
    })
    
    # Sort by date
    result_df = result_df.sort_values('Date')
    
    return result_df

# Example usage
file_path = "C://Users//chena//Desktop//code//backtesterv2//data//Example.xlsx"
try:
    result = process_excel_dates(file_path)
    print(result.head())
    
    # Optionally save to CSV
    result.to_csv('matched_prices.csv', index=False)
    
except Exception as e:
    print(f"Error processing file: {str(e)}")

