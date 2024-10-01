# utils/helpers.py

import matplotlib.pyplot as plt

def plot_portfolio_performance(portfolio):
    plt.figure(figsize=(10, 6))
    plt.plot(portfolio.index, portfolio['total'], label='Portfolio value')
    plt.title('Portfolio Performance')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.show()