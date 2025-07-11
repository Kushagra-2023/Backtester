```markdown
# Backtester: Modular Trading Strategy Engine in Python

This repository provides a customizable backtesting engine for evaluating algorithmic trading strategies using historical financial data. It supports real-time portfolio tracking, custom strategy logic, visualizations, and performance metrics.

## Features

- Supports Yahoo Finance data and CSV input  
- Real-time equity curve, candlestick, and drawdown plots via Plotly  
- Built-in metrics: total return, Sharpe ratio, Sortino ratio, CAGR, and maximum drawdown  
- Easily extendable via a `Strategy` base class  
- Chunked data loading for large datasets

## Architecture

```

├── DataLoader      → Loads & optionally chunks historical data
├── Strategy        → Base class for custom strategies
├── Metrics         → Computes financial performance statistics
├── Plotter         → Generates charts (candlestick, equity curve, drawdown)
└── Engine          → Runs the backtest loop

````

Define your strategy by inheriting from the `Strategy` class and implementing the `init()` and `next()` methods.

## Visualization Outputs

- Candlestick charts  
- Portfolio equity curve  
- Drawdown curve  
- Returns histogram

## Installation

```bash
pip install -r requirements.txt
````

Dependencies include: `pandas`, `yfinance`, `plotly`, `cufflinks`, `seaborn`, `numpy`

## Notebooks

* `example.ipynb`: Try out a sample strategy with live charts.
