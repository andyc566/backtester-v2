# main.py

from data.data_loader import load_spot_prices, load_futures_prices
from data.alternative_data import generate_weather_data, generate_storage_data
from strategies.example_strategies import MovingAverageCrossover
from backtester.backtester import Backtester
from utils.helpers import plot_portfolio_performance

def main():
    # Load data
    spot_prices = load_spot_prices('path_to_your_excel_file.xlsx')
    futures_prices = load_futures_prices('path_to_your_excel_file.xlsx')
    
    # Generate alternative data
    weather_data = generate_weather_data(spot_prices.index.min(), spot_prices.index.max())
    storage_data = generate_storage_data(spot_prices.index.min(), spot_prices.index.max())
    
    # Combine all data
    combined_data = spot_prices.join([futures_prices, weather_data, storage_data])
    
    # Create and run strategy
    strategy = MovingAverageCrossover(combined_data)
    backtester = Backtester(combined_data, strategy)
    results = backtester.run_backtest()
    
    # Plot results
    plot_portfolio_performance(results)

if __name__ == "__main__":
    main()