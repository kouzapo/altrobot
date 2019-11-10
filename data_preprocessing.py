#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

def three_past_closing(dataset):
    #dates = dataset['Date']

    X = dataset.drop(['Volume', 'Open', 'High', 'Low', 'Close'], axis = 1)
    X['Adj Close -1'] = X['Adj Close'].shift(1)
    X['Adj Close -2'] = X['Adj Close'].shift(2)
    #X = X.set_index('Date')
    X = X[2:-1]

    dates = X['Date']

    X = X.drop(['Date'], axis = 1)
    X.index = dates

    y = np.sign((dataset['Adj Close'] - dataset['Adj Close'].shift(1)).dropna())
    y[y == 0] = 1
    y = y.shift(-1).dropna()[1:]
    y.index = X.index
    y.name = 'Sign'

    returns = (dataset['Adj Close'] / dataset['Adj Close'].shift(1) - 1).dropna()
    returns = returns.shift(-1).dropna()[1:]
    returns.index = X.index
    returns.name = 'Returns'

    return X, y, returns