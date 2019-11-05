#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools

import numpy as np
import pandas as pd

from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import style

style.use('ggplot')

class Backtester:
    def __init__(self, X, y, returns, asset_name, model, strategy, portfolio):
        self.X = X
        self.y = y
        self.returns = returns
        self.asset_name = asset_name

        self.model = model
        self.strategy = strategy
        self.portfolio = portfolio

        self.backtest_periods = []
    
    def splitTrainTest(self, by_index = None, by_date = None, single_split = True, window = -1):
        if by_index is not None:
            train_start = by_index['Train'][0]
            train_end = by_index['Train'][1]
            test_start = by_index['Test'][0]
            test_end = by_index['Test'][1]
        
        if by_date is not None:
            train_start = self.X.index.get_loc(by_date['Train'][0])
            train_end = self.X.index.get_loc(by_date['Train'][1]) + 1
            test_start = self.X.index.get_loc(by_date['Test'][0])
            test_end = self.X.index.get_loc(by_date['Test'][1]) + 1
        
        if single_split:
            self.backtest_periods.append({'Train': (train_start, train_end), 'Test': (test_start, test_end)})
        
        else:
            i = train_start
            training_days = train_end - train_start
            
            while i + training_days + window <= test_end:
                self.backtest_periods.append({'Train': (i, i + training_days), 'Test': (i + training_days, i + training_days + window)})
                
                i += window
    
    def __makePredictions(self):
        X = self.X
        y = self.y

        predictions = []

        for P in self.backtest_periods:
            train_i = P['Train']
            test_i = P['Test']

            X_train = X[train_i[0]:train_i[1]]
            y_train = y[train_i[0]:train_i[1]]

            X_test = X[test_i[0]:test_i[1]]
            y_test = y[test_i[0]:test_i[1]]

            self.model.runGridSearch(X_train, y_train)
            predictions.append(self.model.predict(X_test))

            print(P)
        
        return list(itertools.chain.from_iterable(predictions))
        
    def runTest(self):
        start = self.backtest_periods[0]['Test'][0]
        end = self.backtest_periods[-1]['Test'][1]

        y_true = self.y[start:end]
        returns = self.returns[start:end]

        predictions = self.__makePredictions()
        signals = self.strategy.generateSignals(predictions)

        self.portfolio.calcErrorMetrics(predictions, y_true)
        self.portfolio.calcProfitabilityMetrics(signals, returns)

        #print(self.portfolio.error_metrics)
    
    def report(self):
        start = self.backtest_periods[0]['Test'][0]
        end = self.backtest_periods[-1]['Test'][1]

        error_metrics_report = pd.DataFrame([self.portfolio.error_metrics], columns = ['Accuracy', 'Precision', 'Recall', 'F1 Score'], index = [self.model.name])

        print()
        print('Performance metrics for:', self.asset_name)
        print('Testing period: {} to {}'.format(self.X.index[start], self.X.index[end - 1]))
        print()

        print('------------------------Error Metrics-----------------------')
        print(error_metrics_report)
        print()