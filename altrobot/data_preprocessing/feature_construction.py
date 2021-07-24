#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Tuple

import numpy as np
import pandas as pd
import ta


class FeatureConstructor:

    def __init__(
            self,
            dataset: pd.DataFrame,
            training_size: int,
            testing_period: Tuple[str, str] = None,
            train_end: str = None
        ):
        if testing_period:
            train_start = dataset['Date'][dataset['Date'] == testing_period[0]].index[0] - training_size
            self.dates = (dataset['Date'].iloc[train_start], testing_period[1])
        elif train_end:
            train_start = dataset['Date'][dataset['Date'] == train_end].index[0] - training_size
            self.dates = (dataset['Date'].iloc[train_start], train_end)

        self.dataset = dataset

    def _returns(self) -> pd.Series:
        start = self.dates[0]
        end = self.dates[1]

        close = self.dataset['Adj Close']
        close.index = self.dataset['Date']

        returns = close.pct_change().dropna()[start:end]
        returns = returns.shift(-1).dropna()

        return returns

    def _labels(
            self,
            returns: pd.Series
        ) -> pd.Series:
        y = np.sign(returns)
        y[y == 0] = 1
        y[y == -1] = 0

        return y

    def _technical_indicators(self) -> pd.DataFrame:
        close = self.dataset['Adj Close']
        high = self.dataset['High']
        low = self.dataset['Low']

        X = pd.DataFrame()

        X['RSI'] = ta.momentum.RSIIndicator(close).rsi()
        X['MACD'] = ta.trend.MACD(close).macd_diff()
        X['Williams %R'] = ta.momentum.WilliamsRIndicator(high, low, close).williams_r()
        X['Stoc. Osc.'] = ta.momentum.StochasticOscillator(high, low, close, window = 1).stoch()
        X['ROC'] = ta.momentum.ROCIndicator(close, window = 1).roc()
        X['Bol. Bands'] = ta.volatility.BollingerBands(close).bollinger_mavg()
        X['Par. SAR'] = ta.trend.PSARIndicator(high, low, close).psar()
        X['ADX'] = ta.trend.ADXIndicator(high, low, close).adx()

        X.index = self.dataset['Date']

        return X

    def run_preprocessing(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        start = self.dates[0]
        end = self.dates[1]

        features = self._technical_indicators()

        X = features.loc[start:end][:-1]#.values
        returns = self._returns()
        y = self._labels(returns)

        return X, y, returns
