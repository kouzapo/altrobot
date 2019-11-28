#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools

import numpy as np
import pandas as pd

from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import style

from utils import progress_bar

from strategy import AllInOutStrategy
from portfolio import Portfolio

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
    
    def __benchmark_metrics(self):
        start = self.backtest_periods[0]['Test'][0]
        end = self.backtest_periods[-1]['Test'][1]

        y_true = self.y[start:end]
        returns = self.returns[start:end]

        self.bnh_strategy = AllInOutStrategy()
        self.bnh_portfolio = Portfolio()

        predictions = np.ones(len(self.y[start:end]), dtype = int)
        signals = self.bnh_strategy.generate_signals(predictions)

        self.bnh_portfolio.calc_error_metrics(predictions, y_true)
        self.bnh_portfolio.calc_profitability_metrics(signals, returns)

        return self.bnh_portfolio.error_metrics, self.bnh_portfolio.profitability_metrics

    def __make_predictions(self):
        X = self.X
        y = self.y

        predictions = []
        n = len(self.backtest_periods)
        i = 0

        progress_bar(0, n, prefix = 'Backtesting:', length = 50)

        for P in self.backtest_periods:
            train_i = P['Train']
            test_i = P['Test']

            X_train = X[train_i[0]:train_i[1]]
            y_train = y[train_i[0]:train_i[1]]

            X_test = X[test_i[0]:test_i[1]]
            y_test = y[test_i[0]:test_i[1]]

            self.model.grid_search(X_train, y_train)
            predictions.append(self.model.predict(X_test))

            progress_bar(i, n, prefix = 'Backtesting:', length = 50)

            i += 1

        progress_bar(n, n, prefix = 'Backtesting:', length = 50)
        
        return list(itertools.chain.from_iterable(predictions))
    
    def generate_periods(self, split, single_split = True, window = -1):
        train_start = split['Train'][0]
        train_end = split['Train'][1]
        test_start = split['Test'][0]
        test_end = split['Test'][1]
        
        if single_split:
            self.backtest_periods.append({'Train': (train_start, train_end), 'Test': (test_start, test_end)})
        
        else:
            i = train_start
            training_days = train_end - train_start
            
            while i + training_days + window <= test_end:
                self.backtest_periods.append({'Train': (i, i + training_days), 'Test': (i + training_days, i + training_days + window)})
                
                i += window

        self.backtest_periods[-1]['Test'] = (self.backtest_periods[-1]['Test'][0], len(self.X))
    
    def plot_CR(self):
        plt.plot(self.portfolio.cumulative_return, label = 'AAA')
        plt.plot(self.bnh_portfolio.cumulative_return, label = 'Buy & Hold')

        plt.ylabel('Cumulative Return')
        plt.xlabel('Time')
        plt.title('Cumulative Return for {}'.format(self.asset_name))
        plt.legend(loc = 2)

        plt.show()
        
    def test(self):
        start = self.backtest_periods[0]['Test'][0]
        end = self.backtest_periods[-1]['Test'][1]

        y_true = self.y[start:end]
        returns = self.returns[start:end]

        predictions = self.__make_predictions()

        signals = self.strategy.generate_signals(predictions)

        self.__benchmark_metrics()
        self.portfolio.calc_error_metrics(predictions, y_true)
        self.portfolio.calc_profitability_metrics(signals, returns)
    
    def report(self):
        start = self.backtest_periods[0]['Test'][0]
        end = self.backtest_periods[-1]['Test'][1]

        error_metrics_report = pd.DataFrame([self.portfolio.error_metrics], columns = ['Accuracy', 'Precision', 'Recall', 'F1 Score'], index = [self.model.name])
        profitability_metrics_report = pd.DataFrame([self.portfolio.profitability_metrics], columns = ['CR', 'AR', 'AV', 'SR'], index = [self.model.name])

        bnh_error_metrics = pd.DataFrame([self.bnh_portfolio.error_metrics], columns = ['Accuracy', 'Precision', 'Recall', 'F1 Score'], index = ['Buy & Hold'])
        bnh_profitability_metrics = pd.DataFrame([self.bnh_portfolio.profitability_metrics], columns = ['CR', 'AR', 'AV', 'SR'], index = ['Buy & Hold'])

        print()
        print('Performance metrics for:', self.asset_name)
        print('Testing period: {} to {}'.format(self.X.index[start], self.X.index[end - 1]))
        print()

        print('------------------BnH Error Metrics----------------')
        print(bnh_error_metrics)
        print()

        print('------------BnH Profitability Metrics------------')
        print(bnh_profitability_metrics)
        print()

        print('------------------------Error Metrics-----------------------')
        print(error_metrics_report)
        print()

        print('-------------------Profitability Metrics-------------------')
        print(profitability_metrics_report)
        print()

        self.plot_CR()