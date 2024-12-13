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


import requests
import sys
import os

class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._original_stdin = sys.stdin
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        sys.stdin = open(os.devnull, 'r')

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdin.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        sys.stdin = self._original_stdin