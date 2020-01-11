#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools

import numpy as np
import pandas as pd

from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import style

from policy import AllInOutPolicy
from portfolio import BacktestPortfolio
from utils import progress_bar, load_model

style.use('ggplot')

class Backtester:
    def __init__(self, X, y, returns, asset_name, model_names, policy):
        self.X = X
        self.y = y
        self.returns = returns
        self.asset_name = asset_name

        self.model_names = model_names
        self.policy = policy

        self.backtest_periods = []

        #self.portfolio = BacktestPortfolio()
        #self.predictions = []

        #self.portfolios = {model_name: BacktestPortfolio() for model_name in self.models.keys()}
        #self.predictions = {model_name: [] for model_name in self.models.keys()}
    
    def _benchmark_metrics(self):
        start = self.backtest_periods[0]['Test'][0]
        end = self.backtest_periods[-1]['Test'][1]

        y_true = self.y[start:end]
        returns = self.returns[start:end]

        self.bnh_policy = AllInOutPolicy()
        self.bnh_portfolio = BacktestPortfolio()

        predictions = pd.Series(np.ones(len(self.y[start:end]), dtype = int))
        signals = self.bnh_policy.generate_signals(predictions)

        self.bnh_portfolio.calc_error_metrics(predictions, y_true)
        self.bnh_portfolio.calc_profitability_metrics(signals, returns)
    
    def generate_periods(self, training_size, window = -1):
        i = 0
        n = len(self.X)
            
        while i + training_size + window <= n:
            self.backtest_periods.append({'Train': (i, i + training_size), 'Test': (i + training_size, i + training_size + window)})
            
            i += window

        self.backtest_periods[-1]['Test'] = (self.backtest_periods[-1]['Test'][0], len(self.X))
    
    def _predict(self, model_name):
        X = self.X
        y = self.y

        n = len(self.backtest_periods)
        i = 0

        print('Loading {}...'.format(model_name), end = '\r')
        model = load_model(model_name)
        print('Loading {}... Done'.format(model_name))

        print('Compiling {}...'.format(model_name), end = '\r')
        model.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ['accuracy'])
        print('Compiling {}... Done'.format(model_name))

        progress_bar(0, n, prefix = model_name + ':', length = 20)

        for period in self.backtest_periods:
            train_i = period['Train']
            test_i = period['Test']

            X_train = X[train_i[0]:train_i[1]]
            y_train = y[train_i[0]:train_i[1]]

            X_test = X[test_i[0]:test_i[1]]

            X_train = StandardScaler().fit_transform(X_train)
            X_test = StandardScaler().fit_transform(X_test)

            model.fit(X_train, y_train, batch_size = 100, epochs = 100, verbose = 0)

            predicted_probs = model.predict(X_test)[:, 0]
            P = [1 if p >= 0.5 else 0 for p in predicted_probs]

            self.predictions[model_name].append(P)

            progress_bar(i, n, prefix = model_name + ':', length = 20)
            i += 1
        
        progress_bar(n, n, prefix = model_name + ':', length = 20)
        print()

        self.predictions[model_name] = pd.Series(list(itertools.chain.from_iterable(self.predictions[model_name])))
    
    def plot_CR(self):
        plt.plot(self.bnh_portfolio.cumulative_return, label = 'Buy & Hold')

        for model_name, portfolio in self.portfolios.items():
            plt.plot(portfolio.cumulative_return, label = model_name)

        plt.ylabel('Cumulative Return')
        plt.xlabel('Time')
        plt.title('Cumulative Return for {}'.format(self.asset_name))
        plt.legend(loc = 2)

        plt.show()
    
    def test(self, n):
        start = self.backtest_periods[0]['Test'][0]
        end = self.backtest_periods[-1]['Test'][1]

        y_true = self.y[start:end]
        returns = self.returns[start:end]

        self._benchmark_metrics()

        print('Training {} model(s) {} time(s) each:\n'.format(len(self.model_names), n))

        for i in range(n):
            self.portfolios = {model_name: BacktestPortfolio() for model_name in self.model_names}
            self.predictions = {model_name: [] for model_name in self.model_names}

            for model_name in self.model_names:
                self._predict(model_name)

                signals = self.policy.generate_signals(self.predictions[model_name])

                self.portfolios[model_name].calc_error_metrics(self.predictions[model_name], y_true)
                self.portfolios[model_name].calc_profitability_metrics(signals, returns)

            print()
            
            error_metrics_report = pd.DataFrame([self.bnh_portfolio.error_metrics] + [self.portfolios[model_name].error_metrics for model_name in self.model_names], columns = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'PT p-value'], index = ['Buy & Hold'] + [model_name for model_name in self.model_names])
            profitability_metrics_report = pd.DataFrame([self.bnh_portfolio.profitability_metrics] + [self.portfolios[model_name].profitability_metrics for model_name in self.model_names], columns = ['CR', 'AR', 'AV', 'SR'], index = ['Buy & Hold'] + [model_name for model_name in self.model_names])

            error_metrics_report.index.name = 'Model name'
            profitability_metrics_report.index.name = 'Model name'
            
            error_metrics_report.to_csv('backtest_results/' + self.asset_name + '_acc_' + str(i) + '.csv')
            profitability_metrics_report.to_csv('backtest_results/' + self.asset_name + '_prof_' + str(i) + '.csv')
    
    def report(self, n):
        start = self.backtest_periods[0]['Test'][0]
        end = self.backtest_periods[-1]['Test'][1]

        acc_concat = pd.concat([pd.read_csv('backtest_results/' + self.asset_name + '_acc_' + str(i) + '.csv', index_col = 'Model name') for i in range(n)])
        acc_groupby = acc_concat.groupby(acc_concat.index)

        perf_concat = pd.concat([pd.read_csv('backtest_results/' + self.asset_name + '_prof_' + str(i) + '.csv', index_col = 'Model name') for i in range(n)])
        perf_groupby = perf_concat.groupby(perf_concat.index)

        error_metrics_report = acc_groupby.mean()
        profitability_metrics_report = perf_groupby.mean()





        print('\n===========Performance metrics for {}==========='.format(self.asset_name))
        print('Testing period: {} - {}'.format(self.X.index[start], self.X.index[end - 1]))
        print('Models tested: {}\n'.format(len(self.model_names)))

        print('---------------------------Error metrics-------------------------')
        print(error_metrics_report)
        print()

        print('------------------Profitability metrics--------------')
        print(profitability_metrics_report)
        print()

        error_metrics_report.to_csv('backtest_results/' + self.asset_name + '_acc.csv')
        profitability_metrics_report.to_csv('backtest_results/' + self.asset_name + '_prof.csv')

        self.plot_CR()