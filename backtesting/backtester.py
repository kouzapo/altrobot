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

from policy import AllInOutPolicy
from portfolio import BacktestPortfolio
from utils import progress_bar

style.use('ggplot')

class Backtester:
    def __init__(self, X, y, returns, asset_name, models, policy):
        self.X = X
        self.y = y
        self.returns = returns
        self.asset_name = asset_name

        self.models = models
        self.policy = policy

        self.backtest_periods = []

        #self.portfolio = BacktestPortfolio()
        #self.predictions = []

        self.portfolios = {model_name: BacktestPortfolio() for model_name in self.models.keys()}
        self.predictions = {model_name: [] for model_name in self.models.keys()}
    
    def _benchmark_metrics(self):
        start = self.backtest_periods[0]['Test'][0]
        end = self.backtest_periods[-1]['Test'][1]

        y_true = self.y[start:end]
        returns = self.returns[start:end]

        self.bnh_policy = AllInOutPolicy()
        self.bnh_portfolio = BacktestPortfolio()

        predictions = np.ones(len(self.y[start:end]), dtype = int)
        signals = self.bnh_policy.generate_signals(predictions)

        self.bnh_portfolio.calc_error_metrics(predictions, y_true)
        self.bnh_portfolio.calc_profitability_metrics(signals, returns)
    
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
    
    def _predict(self, model_name):
        X = self.X
        y = self.y

        n = len(self.backtest_periods)
        i = 0

        progress_bar(0, n, prefix = model_name + ':', length = 20)

        for period in self.backtest_periods:
            train_i = period['Train']
            test_i = period['Test']

            X_train = X[train_i[0]:train_i[1]]
            y_train = y[train_i[0]:train_i[1]]

            X_test = X[test_i[0]:test_i[1]]
            y_test = y[test_i[0]:test_i[1]]

            X_train = StandardScaler().fit_transform(X_train)
            X_test = StandardScaler().fit_transform(X_test)

            self.models[model_name].fit(X_train, y_train, batch_size = 100, epochs = 100, verbose = 0)

            #predictions = self.model.predict(X_test)[:, 0]
            predicted_probs = self.models[model_name].predict(X_test)[:, 0]
            P = [1 if p >= 0.5 else 0 for p in predicted_probs]

            self.predictions[model_name].append(P)

            progress_bar(i, n, prefix = model_name + ':', length = 20)
            i += 1
        
        progress_bar(n, n, prefix = model_name + ':', length = 20)

        self.predictions[model_name] = list(itertools.chain.from_iterable(self.predictions[model_name]))
    
    def plot_CR(self):
        plt.plot(self.bnh_portfolio.cumulative_return, label = 'Buy & Hold')

        for model_name, portfolio in self.portfolios.items():
            plt.plot(portfolio.cumulative_return, label = model_name)

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

        print('Training {} model(s):\n'.format(len(self.models)))
        
        for model_name in self.models.keys():
            #print(self.models[model_name])
            self._predict(model_name)

            signals = self.policy.generate_signals(self.predictions[model_name])

            self.portfolios[model_name].calc_error_metrics(self.predictions[model_name], y_true)
            self.portfolios[model_name].calc_profitability_metrics(signals, returns)

        self._benchmark_metrics()
    
    def report(self):
        start = self.backtest_periods[0]['Test'][0]
        end = self.backtest_periods[-1]['Test'][1]

        error_metrics_report = pd.DataFrame([self.bnh_portfolio.error_metrics] + [self.portfolios[model_name].error_metrics for model_name in self.portfolios.keys()], columns = ['Accuracy', 'Precision', 'Recall', 'F1 Score'], index = ['Buy & Hold'] + [model_name for model_name in self.models.keys()])
        profitability_metrics_report = pd.DataFrame([self.bnh_portfolio.profitability_metrics] + [self.portfolios[model_name].profitability_metrics for model_name in self.portfolios.keys()], columns = ['CR', 'AR', 'AV', 'SR'], index = ['Buy & Hold'] + [model_name for model_name in self.models.keys()])

        print('\n\n===========Performance metrics for {}==========='.format(self.asset_name))
        print('Testing period: {} to {}'.format(self.X.index[start], self.X.index[end - 1]))
        print('Models tested: {}\n'.format(len(self.models)))

        print('--------------------Error metrics------------------')
        print(error_metrics_report)
        print()

        print('----------------Profitability metrics--------------')
        print(profitability_metrics_report)
        print()

        self.plot_CR()