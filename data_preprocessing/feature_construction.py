#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import ta

class FeatureConstructor:
    def __init__(self, dataset, dates):
        self.dataset = dataset
        self.dates = dates
    
    def _returns(self):
        start = self.dates[0]
        end = self.dates[1]

        close = self.dataset['Adj Close']
        close.index = self.dataset['Date']

        returns = close.pct_change().dropna()[start:end]
        returns = returns.shift(-1).dropna()

        return returns
    
    def _labels(self, returns):
        y = np.sign(returns)
        y[y == 0] = 1
        y[y == -1] = 0

        return y
    
    def _three_past_closing(self):
        start = self.dates[0]
        end = self.dates[1]

        start_i = self.dataset['Date'][self.dataset['Date'] == start].index[0]
        end_i = self.dataset['Date'][self.dataset['Date'] == end].index[0]

        X = pd.DataFrame()

        X['Adj Close'] = self.dataset['Adj Close']
        X['Adj Close -1'] = self.dataset['Adj Close'].shift(1)
        X['Adj Close -2'] = self.dataset['Adj Close'].shift(2)

        X = X[start_i:end_i]
        X.index = self.dataset['Date'][start_i:end_i]

        return X
    
    def _technical_indicators(self):
        start = self.dates[0]
        end = self.dates[1]

        close = self.dataset['Adj Close']
        high = self.dataset['High']
        low = self.dataset['Low']

        close.index = self.dataset['Date']
        high.index = self.dataset['Date']
        low.index = self.dataset['Date']

        X = pd.DataFrame()

        X['RSI'] = ta.momentum.RSIIndicator(close).rsi()
        X['MACD'] = ta.trend.MACD(close).macd_diff()
        X['Williams %R'] = ta.momentum.WilliamsRIndicator(high, low, close).wr()
        X['Stoc. Osc.'] = ta.momentum.StochasticOscillator(high, low, close, d_n = 1).stoch()
        #X['Aroon'] = ta.trend.AroonIndicator(close).aroon_indicator()
        X['ROC'] = ta.momentum.ROCIndicator(close, n = 1).roc()
        X['Bol. Bands'] = ta.volatility.BollingerBands(close).bollinger_mavg()
        X['Par. SAR'] = ta.trend.PSARIndicator(high, low, close).psar()
        X['ADX'] = ta.trend.ADXIndicator(high, low, close).adx()

        X = X.loc[start:end]

        return X

    def run_preprocessing(self):
        #X = self._three_past_closing()
        X = self._technical_indicators()[:-1]

        returns = self._returns()
        y = self._labels(returns)

        return X, y, returns