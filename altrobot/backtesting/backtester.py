#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from matplotlib import style

from policy import AllInOutPolicy
from backtesting.portfolio import BacktestPortfolio
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
    
    def _benchmark_metrics(self):
        start = self.backtest_periods[0]['test'][0]
        end = self.backtest_periods[-1]['test'][1]

        y_true = self.y[start:end]
        returns = self.returns[start:end]

        self.bnh_portfolio = BacktestPortfolio()

        predictions = pd.Series(np.ones(len(y_true), dtype = int), index = self.y.index[start:end])
        signals = self.policy.generate_signals(predictions)

        self.bnh_portfolio.calc_error_metrics(predictions, y_true)
        self.bnh_portfolio.calc_profitability_metrics(signals, returns)
        self.bnh_portfolio.calc_conf_matrix(predictions, y_true)
        self.bnh_portfolio.calc_conf_matrix_prof(predictions, y_true, returns)
    
    def generate_periods(self, training_size, window):
        i = 0
        n = len(self.X)
            
        while i + training_size + window <= n:
            self.backtest_periods.append({'train': (i, i + training_size), 'test': (i + training_size, i + training_size + window)})
            
            i += window

        self.backtest_periods[-1]['test'] = (self.backtest_periods[-1]['test'][0], len(self.X))
    
    def _write_reports(self, n):
        err_concat = pd.concat([pd.read_csv(f'backtest_results/{self.asset_name}_err_{str(i)}.csv', index_col = 'Model name') for i in range(n)])
        err_groupby = err_concat.groupby(err_concat.index)

        perf_concat = pd.concat([pd.read_csv(f'backtest_results/{self.asset_name}_prof_{str(i)}.csv', index_col = 'Model name') for i in range(n)])
        perf_groupby = perf_concat.groupby(perf_concat.index)

        conf_matrix_concat = pd.concat([pd.read_csv(f'backtest_results/{self.asset_name}_conf_mat_{str(i)}.csv', index_col = 'Model name') for i in range(n)])
        conf_matrix_groupby = conf_matrix_concat.groupby(conf_matrix_concat.index)

        conf_matrix_prof_concat = pd.concat([pd.read_csv(f'backtest_results/{self.asset_name}_conf_mat_prof_{str(i)}.csv', index_col = 'Model name') for i in range(n)])
        conf_matrix_prof_groupby = conf_matrix_prof_concat.groupby(conf_matrix_prof_concat.index)

        error_metrics_report = err_groupby.mean()
        profitability_metrics_report = perf_groupby.mean()
        confusion_matrix_report = conf_matrix_groupby.mean()
        confusion_matrix_prof_report = conf_matrix_prof_groupby.mean()

        error_metrics_report.to_csv(f'backtest_results/{self.asset_name}_err.csv')
        profitability_metrics_report.to_csv(f'backtest_results/{self.asset_name}_prof.csv')
        confusion_matrix_report.to_csv(f'backtest_results/{self.asset_name}_conf_mat.csv')
        confusion_matrix_prof_report.to_csv(f'backtest_results/{self.asset_name}_conf_mat_prof.csv')
    
    def _predict(self, model_name):
        start = self.backtest_periods[0]['test'][0]
        end = self.backtest_periods[-1]['test'][1]
        n = len(self.backtest_periods)
        i = 0

        print('Loading {}...'.format(model_name), end = '\r')
        model = load_model(model_name)
        model.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ['accuracy'])
        print('Loading {}... Done'.format(model_name))

        progress_bar(0, n, prefix = f'{model_name}:', length = 20)

        for period in self.backtest_periods:
            train_i = period['train']
            test_i = period['test']

            X_train = self.X[train_i[0]:train_i[1]]
            y_train = self.y[train_i[0]:train_i[1]]

            X_test = self.X[test_i[0]:test_i[1]]

            X_train = StandardScaler().fit_transform(X_train)
            X_test = StandardScaler().fit_transform(X_test)

            model.fit(X_train, y_train, batch_size = 100, epochs = 100, verbose = 0)

            predicted_probs = model.predict(X_test)[:, 0]

            self.predicted_probs[model_name].extend(list(predicted_probs))
            self.predictions[model_name].extend([1 if p >= 0.5 else 0 for p in predicted_probs])

            progress_bar(i, n, prefix = f'{model_name}:', length = 20)
            i += 1
        
        progress_bar(n, n, prefix = f'{model_name}:', length = 20)
        print()

        self.predictions[model_name] = pd.Series(self.predictions[model_name], index = self.y.index[start:end])
   
    def plot_CR(self):
        plt.plot(self.bnh_portfolio.cumulative_return, label = 'Buy & Hold')

        for model_name, portfolio in self.portfolios.items():
            plt.plot(portfolio.cumulative_return, label = model_name)

        plt.ylabel('Cumulative Return')
        plt.xlabel('Time')
        plt.title(f'Cumulative Return for {self.asset_name}')
        plt.legend(loc = 2)

        plt.show()
    
    def test(self, n):
        start = self.backtest_periods[0]['test'][0]
        end = self.backtest_periods[-1]['test'][1]

        y_true = self.y[start:end]
        returns = self.returns[start:end]

        if not os.path.isdir('backtest_results/'):
            os.mkdir('backtest_results/')

        self._benchmark_metrics()

        print(f'Training {len(self.model_names)} model(s) {n} time(s) each:\n')

        for i in range(n):
            self.portfolios = {model_name: BacktestPortfolio() for model_name in self.model_names}
            self.predictions = {model_name: [] for model_name in self.model_names}
            self.predicted_probs = {model_name: [] for model_name in self.model_names}

            for model_name in self.model_names:
                self._predict(model_name)

                signals = self.policy.generate_signals(self.predictions[model_name])

                self.portfolios[model_name].calc_error_metrics(self.predictions[model_name], y_true)
                self.portfolios[model_name].calc_profitability_metrics(signals, returns, self.bnh_portfolio.annualized_return)
                self.portfolios[model_name].calc_conf_matrix(self.predictions[model_name], y_true)
                self.portfolios[model_name].calc_conf_matrix_prof(self.predictions[model_name], y_true, returns)

            print()
            
            error_metrics_report = pd.DataFrame([self.bnh_portfolio.error_metrics] + [self.portfolios[model_name].error_metrics for model_name in self.model_names], columns = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'PT p-value'], index = ['Buy & Hold'] + self.model_names)
            profitability_metrics_report = pd.DataFrame([self.bnh_portfolio.profitability_metrics] + [self.portfolios[model_name].profitability_metrics for model_name in self.model_names], columns = ['CR', 'AR', 'AV', 'SR', 'IR'], index = ['Buy & Hold'] + self.model_names)
            conf_matrix_report = pd.DataFrame([self.bnh_portfolio.conf_matrix] + [self.portfolios[model_name].conf_matrix for model_name in self.model_names], columns = ['TP', 'TN', 'FP', 'FN'], index = ['Buy & Hold'] + self.model_names)
            conf_matrix_prof_report = pd.DataFrame([self.bnh_portfolio.conf_matrix_prof] + [self.portfolios[model_name].conf_matrix_prof for model_name in self.model_names], columns = ['TP', 'TN', 'FP', 'FN'], index = ['Buy & Hold'] + self.model_names)            

            error_metrics_report.index.name = 'Model name'
            profitability_metrics_report.index.name = 'Model name'
            conf_matrix_report.index.name = 'Model name'
            conf_matrix_prof_report.index.name = 'Model name'
            
            error_metrics_report.to_csv(f'backtest_results/{self.asset_name}_err_{str(i)}.csv')
            profitability_metrics_report.to_csv(f'backtest_results/{self.asset_name}_prof_{str(i)}.csv')
            conf_matrix_report.to_csv(f'backtest_results/{self.asset_name}_conf_mat_{str(i)}.csv')
            conf_matrix_prof_report.to_csv(f'backtest_results/{self.asset_name}_conf_mat_prof_{str(i)}.csv')
        
        self._write_reports(n)
    
    def report(self):
        start = self.backtest_periods[0]['test'][0]
        end = self.backtest_periods[-1]['test'][1]

        error_metrics_report = pd.read_csv(f'backtest_results/{self.asset_name}_err.csv', index_col = 'Model name')
        profitability_metrics_report = pd.read_csv(f'backtest_results/{self.asset_name}_prof.csv', index_col = 'Model name')
        confusion_matrix_report = pd.read_csv(f'backtest_results/{self.asset_name}_conf_mat.csv', index_col = 'Model name')
        confusion_matrix_prof_report = pd.read_csv(f'backtest_results/{self.asset_name}_conf_mat_prof.csv', index_col = 'Model name')





        print(f'\n===========Performance metrics for {self.asset_name}===========')
        print(f'Testing period: {self.y.index[start]} - {self.y.index[end - 1]}')
        print(f'Models tested: {len(self.model_names)}\n')

        print('---------------------------Error metrics-------------------------')
        print(error_metrics_report)
        print()

        print('-----------------------Profitability metrics-------------------')
        print(profitability_metrics_report)
        print()

        print('-------------Confusion matrix-----------')
        print(confusion_matrix_report)
        print()

        print('-------------Confusion matrix performance-----------')
        print(confusion_matrix_prof_report)
        print()

        #self.plot_CR()