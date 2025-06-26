import sys
import pandas as pd
import yfinance as yf
from typing import *
import datetime
from datetime import *
from abc import ABC, abstractmethod
import numpy as np
import chart_studio.plotly as py
import cufflinks as cf
import plotly.express as px
import seaborn as sns
import plotly.graph_objects as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected = True)
cf.go_offline()

pd.options.plotting.backend = "plotly"

import plotly.io as pio
pio.renderers.default = 'iframe'
import plotly.graph_objects as go

class DataLoader:
    def __init__(self, ticker, start_date, end_date, interval, chunk_size = None, csv_file = None):
        self.ticker = ticker
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.interval = interval
        self.chunk_size = chunk_size
        self.csv_file = csv_file

    def __iter__(self):
        if self.csv_file is not None:
            if self.chunk_size is None:
                yield self.clean_data(pd.read_csv(self.csv_file))
            else:
                for chunk in pd.read_csv(self.csv_file, chunksize=self.chunk_size):
                    yield self.clean_data(chunk)
        else:
            if self.chunk_size is None:
                data = yf.download(self.ticker, start=self.start_date.strftime("%Y-%m-%d"),
                                   end=self.end_date.strftime("%Y-%m-%d"), interval=self.interval)
                yield self.clean_data(data)
            else:
                current_start = self.start_date
                while current_start <= self.end_date:
                    current_end = min(current_start + timedelta(days=self.chunk_size), self.end_date)
                    data = yf.download(self.ticker, start=current_start.strftime("%Y-%m-%d"),
                                       end=current_end.strftime("%Y-%m-%d"), interval=self.interval)
                    yield self.clean_data(data)
                    current_start = current_end

    @staticmethod
    def clean_data(df):
        df.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
        df = df.ffill()
        return df      

class Strategy:
    def __init__(self, data, initial_cash, apply_commission=False, commission_rate=0.001):
        self.data = data
        self.cash = initial_cash
        self.position = 0
        self.apply_commission = apply_commission
        self.commission_rate = commission_rate
        self.current_row = None
        self.date = None
        self.time = None

        print("Strategy initialized. Starting cash:", self.cash)

    def __str__(self):
        return "<Strategy " + str(self) + ">"

    def get_portfolio_value(self):
        current_price = self.current_row['Close']
        return self.cash + self.position * current_price

    def _apply_commission(self, price, is_buy=True):
        if self.apply_commission:
            factor = (1 + self.commission_rate) if is_buy else (1 - self.commission_rate)
            return price * factor
        return price

    def _buy(self, quantity):
        price = quantity * self.current_row['Close']
        price = self._apply_commission(price, is_buy=True)
        if self.cash >= price:
            self.cash -= price
            self.position += quantity
            
            print(f"Bought quantity: {quantity} at total price: {price}")
            
            return True
            
        print(f"Trade not executed, amount insufficient!")
        return False

    def _sell(self, quantity):
        if self.position >= quantity:
            price = quantity * self.current_row['Close']
            price = self._apply_commission(price, is_buy=False)
            self.cash += price
            self.position -= quantity

            print(f"Sold quantity: {quantity} at total price: {price}")
            
            return True
            
        print(f"Trade not executed, quantity greater than available!")
        return False

    def _data(self):
        for idx, row in self.data.iterrows():
            self.date = idx.date()
            self.time = idx.time()
            yield row

    def set_current_data(self):
        if not hasattr(self, 'data_gen'):
            self.data_gen = self._data()  # create generator only once
        try:
            self.current_row = next(self.data_gen)
        except StopIteration:
            self.current_row = None  # End of data


    def buy(self, quantity = 1):
        return self._buy(quantity)

    def sell(self, quantity = 1):
        return self._sell(quantity)

    @abstractmethod
    def init(self):
        pass
        
    @abstractmethod
    def next(self):
        pass

class Metrics:
    def __init__(self, initial_cash):
        self.initial_cash = initial_cash
        self.portfolio_values = [] # to be set in Engine's run
        
    def append_portfolio_value(self, val):
        self.portfolio_values.append(val)
        
    def total_return(self):
        return (self.portfolio_values[-1] - self.initial_cash) / self.initial_cash * 100

    def maximum_drawdown(self):
        pv = np.array(self.portfolio_values)
        peak = np.maximum.accumulate(pv)
        drawdown = (peak - pv) / peak
        return np.max(drawdown)

    def volatility(self):
        returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
        return np.std(returns)

    def cagr(self):
        total_return = self.portfolio_values[-1] / self.initial_cash
        periods = len(self.portfolio_values)
        
        if self.frequency == 'daily':
            years = periods / 252  # assuming 252 trading days
        elif self.frequency == 'hourly':
            years = periods / (252 * 6.5)  # 6.5 hours/day
        else:
            years = 1  # fallback

        return (total_return ** (1 / years)) - 1

    def sharpe_ratio(self):
        returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
        excess_returns = returns
        return np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 1e-9

    def sortino_ratio(self):
        returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
        downside_returns = [r for r in returns if r < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-9  # avoid div/0
        return np.mean(returns) / downside_std

    def calmar_ratio(self):
        mdd = self.max_drawdown()
        cagr = self.cagr()
        return cagr / mdd if mdd > 0 else np.inf

class Plotter:
    def __init__(self, data):
        self.data = data
        self.portfolio_values = []  # To be set in Engine's run
        self.trade_indices = None
        for df in self.data:
            self.trade_indices = df.index

    def append_portfolio_value(self, val):
        self.portfolio_values.append(val)

    def candlestick(self):
        for df in self.data:
            fig = go.Figure(data=[go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close']
            )])
            fig.update_layout(title='Candlestick Chart', xaxis_title='Time', yaxis_title='Price')
            fig.show()

    def equity_curve(self):
        if not self.portfolio_values:
            print("No portfolio values to plot.")
            return
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=self.portfolio_values,
            x=self.trade_indices,
            mode='lines',
            name='Equity Curve',
        ))

        fig.update_layout(
            title='Equity Curve (Portfolio Value Over Time)',
            xaxis_title='Time',
            yaxis_title='Portfolio Value',
        )
        fig.show()

    def drawdown_curve(self):
        if not self.portfolio_values:
            print("No portfolio values to plot.")
            return
        
        portfolio = pd.Series(self.portfolio_values, index=self.trade_indices)
        running_max = portfolio.cummax()
        drawdown = (portfolio - running_max) / running_max

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.trade_indices,
            y=drawdown,
            mode='lines',
            name='Drawdown'
        ))

        fig.update_layout(
            title='Drawdown Curve',
            xaxis_title='Time',
            yaxis_title='Drawdown (Fraction)',
        )
        fig.show()

    def returns_histogram(self):
        if len(self.portfolio_values) < 2:
            print("Not enough data to compute returns.")
            return

        portfolio = pd.Series(self.portfolio_values, index=self.trade_indices)
        returns = portfolio.pct_change().dropna()

        fig = px.histogram(returns, nbins=50, title='Returns Histogram')
        fig.update_layout(
            xaxis_title='Returns',
            yaxis_title='Frequency'
        )
        fig.show()

class Engine:
    def __init__(self, dataloader, strategy_class, initial_cash, apply_commission=False, commission_rate=0.001):
        self.dataloader = dataloader
        self.initial_cash = initial_cash
        self.current_cash = initial_cash
        self.apply_commission = apply_commission
        self.commission_rate = commission_rate
        self.strategy_class = strategy_class
        
        self.plotter = Plotter(dataloader)
        
    def run(self):
        for data in self.dataloader:
            self.strategy = self.strategy_class(
                data,
                self.current_cash,
                apply_commission=self.apply_commission,
                commission_rate=self.commission_rate
            )
            self.strategy.init()
            metrics = Metrics(self.current_cash)
    
            while True:
                self.strategy.set_current_data()
                
                if self.strategy.current_row is None:
                    break
                    
                metrics.append_portfolio_value(self.strategy.get_portfolio_value())
                self.plotter.append_portfolio_value(self.strategy.get_portfolio_value())
                
                print(f"{self.strategy.date} {self.strategy.time} | Close: {self.strategy.current_row['Close']} | Position: {self.strategy.position} | Cash: {self.strategy.cash}")
                
                self.strategy.next()

    
            self.current_cash = self.strategy.cash
