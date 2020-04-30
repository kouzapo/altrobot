#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import ta

class FeatureConstructor:
    def __init__(self, dataset, testing_period, training_size):
        train_start = dataset['Date'][dataset['Date'] == testing_period[0]].index[0] - training_size

        self.dataset = dataset
        self.dates = (dataset['Date'].iloc[train_start], testing_period[1])
        
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
        X['ROC'] = ta.momentum.ROCIndicator(close, n = 1).roc()
        X['Bol. Bands'] = ta.volatility.BollingerBands(close).bollinger_mavg()
        X['Par. SAR'] = ta.trend.PSARIndicator(high, low, close).psar()
        X['ADX'] = ta.trend.ADXIndicator(high, low, close).adx()

        return X

    def run_preprocessing(self):
        start = self.dates[0]
        end = self.dates[1]

        features = self._technical_indicators()

        X = features.loc[start:end][:-1].values

        returns = self._returns()
        y = self._labels(returns)

        return X, y, returns