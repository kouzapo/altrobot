#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import style

from backtesting.portfolio import BacktestPortfolio
from backtesting.policy import AllInOutPolicy
from utils import progress_bar, load_model

style.use('ggplot')


class Backtester:

    RESULTS_PATH = 'resources/backtest_results/'

    def __init__(self, backtest_subsets, asset_name, model_names, policy):
        self.backtest_subsets = backtest_subsets
        self.asset_name = asset_name

        self.model_names = model_names
        self.policy = policy

        self.y_true = []
        self.returns = []
        self.index = []

        for subset in self.backtest_subsets:
            self.y_true.extend(list(subset['y_test']))
            self.returns.extend(list(subset['returns_test']))
            self.index.extend(list(subset['y_test'].index))

        self.y_true = pd.Series(self.y_true, index = self.index)
        self.returns = pd.Series(self.returns, index = self.index)

    def _benchmark_metrics(self):
        self.bnh_portfolio = BacktestPortfolio()

        predictions = pd.Series(np.ones(len(self.y_true), dtype = int), index = self.index)
        benchmark_policy = AllInOutPolicy(bounds = (0.5, 0.5))
        signals = benchmark_policy.generate_signals(predictions)

        self.bnh_portfolio.calc_error_metrics(predictions, self.y_true)
        self.bnh_portfolio.calc_profitability_metrics(signals, self.returns)
        self.bnh_portfolio.calc_conf_matrix(predictions, self.y_true)
        self.bnh_portfolio.calc_conf_matrix_prof(predictions, self.y_true, self.returns)

    def _export_aggregated_reports(self, n):
        err_concat = pd.concat([pd.read_csv(f'{self.RESULTS_PATH + self.asset_name}_err_{str(i)}.csv',
                                            index_col = 'Model name') for i in range(n)])

        perf_concat = pd.concat([pd.read_csv(f'{self.RESULTS_PATH + self.asset_name}_prof_{str(i)}.csv',
                                            index_col = 'Model name') for i in range(n)])

        conf_matrix_concat = pd.concat([pd.read_csv(f'{self.RESULTS_PATH + self.asset_name}_conf_mat_{str(i)}.csv',
                                            index_col = 'Model name') for i in range(n)])

        conf_matrix_prof_concat = pd.concat([pd.read_csv(f'{self.RESULTS_PATH + self.asset_name}_conf_mat_prof_{str(i)}.csv',
                                            index_col = 'Model name') for i in range(n)])

        error_metrics_report = err_concat.groupby(err_concat.index).mean()
        profitability_metrics_report = perf_concat.groupby(perf_concat.index).mean()
        confusion_matrix_report = conf_matrix_concat.groupby(conf_matrix_concat.index).mean()
        confusion_matrix_prof_report = conf_matrix_prof_concat.groupby(conf_matrix_prof_concat.index).mean()

        error_metrics_report.to_csv(f'{self.RESULTS_PATH + self.asset_name}_err.csv')
        profitability_metrics_report.to_csv(f'{self.RESULTS_PATH + self.asset_name}_prof.csv')
        confusion_matrix_report.to_csv(f'{self.RESULTS_PATH + self.asset_name}_conf_mat.csv')
        confusion_matrix_prof_report.to_csv(f'{self.RESULTS_PATH + self.asset_name}_conf_mat_prof.csv')

    def _predict(self, model_name):
        n = len(self.backtest_subsets)
        i = 0

        print('Loading {}...'.format(model_name), end = '\r')
        model = load_model(model_name)
        model.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ['accuracy'])
        print('Loading {}... Done'.format(model_name))

        progress_bar(0, n, prefix = f'{model_name}:', length = 20)

        for subset in self.backtest_subsets:
            model.fit(subset['X_train'], subset['y_train'], batch_size = 100, epochs = 100, verbose = 0)

            predicted_probs = model.predict(subset['X_test'])[:, 0]

            self.predicted_probs[model_name].extend(list(predicted_probs))
            self.predictions[model_name].extend([1 if p >= 0.5 else 0 for p in predicted_probs])

            progress_bar(i, n, prefix = f'{model_name}:', length = 20)
            i += 1

        progress_bar(n, n, prefix = f'{model_name}:', length = 20)
        print()

        self.predicted_probs[model_name] = pd.Series(self.predicted_probs[model_name], index = self.index)
        self.predictions[model_name] = pd.Series(self.predictions[model_name], index = self.index)

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
        if not os.path.isdir(self.RESULTS_PATH):
            os.mkdir(self.RESULTS_PATH)

        self._benchmark_metrics()

        print(f'Training {len(self.model_names)} model(s) {n} time(s) each:\n')

        for i in range(n):
            self.portfolios = {model_name: BacktestPortfolio() for model_name in self.model_names}
            self.predictions = {model_name: [] for model_name in self.model_names}
            self.predicted_probs = {model_name: [] for model_name in self.model_names}

            for model_name in self.model_names:
                self._predict(model_name)

                signals = self.policy.generate_signals(self.predicted_probs[model_name])

                self.portfolios[model_name].calc_error_metrics(self.predictions[model_name], self.y_true)
                self.portfolios[model_name].calc_profitability_metrics(signals, self.returns, self.bnh_portfolio.annualized_return)
                self.portfolios[model_name].calc_conf_matrix(self.predictions[model_name], self.y_true)
                self.portfolios[model_name].calc_conf_matrix_prof(self.predictions[model_name], self.y_true, self.returns)

            print()

            error_metrics_report = pd.DataFrame([self.bnh_portfolio.error_metrics]
                                    + [self.portfolios[model_name].error_metrics for model_name in self.model_names],
                                    columns = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'PT p-value'],
                                    index = ['Buy & Hold'] + self.model_names)

            profitability_metrics_report = pd.DataFrame([self.bnh_portfolio.profitability_metrics]
                                    + [self.portfolios[model_name].profitability_metrics for model_name in self.model_names],
                                    columns = ['CR', 'AR', 'AV', 'SR', 'IR'], 
                                    index = ['Buy & Hold'] + self.model_names)

            conf_matrix_report = pd.DataFrame([self.bnh_portfolio.conf_matrix]
                                    + [self.portfolios[model_name].conf_matrix for model_name in self.model_names], 
                                    columns = ['TP', 'TN', 'FP', 'FN'], 
                                    index = ['Buy & Hold'] + self.model_names)

            conf_matrix_prof_report = pd.DataFrame([self.bnh_portfolio.conf_matrix_prof]
                                    + [self.portfolios[model_name].conf_matrix_prof for model_name in self.model_names], 
                                    columns = ['TP', 'TN', 'FP', 'FN'], 
                                    index = ['Buy & Hold'] + self.model_names)            

            error_metrics_report.index.name = 'Model name'
            profitability_metrics_report.index.name = 'Model name'
            conf_matrix_report.index.name = 'Model name'
            conf_matrix_prof_report.index.name = 'Model name'

            error_metrics_report.to_csv(f'{self.RESULTS_PATH + self.asset_name}_err_{str(i)}.csv')
            profitability_metrics_report.to_csv(f'{self.RESULTS_PATH + self.asset_name}_prof_{str(i)}.csv')
            conf_matrix_report.to_csv(f'{self.RESULTS_PATH + self.asset_name}_conf_mat_{str(i)}.csv')
            conf_matrix_prof_report.to_csv(f'{self.RESULTS_PATH + self.asset_name}_conf_mat_prof_{str(i)}.csv')

        self._export_aggregated_reports(n)

    def report(self):
        error_metrics_report = pd.read_csv(f'{self.RESULTS_PATH + self.asset_name}_err.csv', index_col = 'Model name')
        profitability_metrics_report = pd.read_csv(f'{self.RESULTS_PATH + self.asset_name}_prof.csv', index_col = 'Model name')
        confusion_matrix_report = pd.read_csv(f'{self.RESULTS_PATH + self.asset_name}_conf_mat.csv', index_col = 'Model name')
        confusion_matrix_prof_report = pd.read_csv(f'{self.RESULTS_PATH + self.asset_name}_conf_mat_prof.csv', index_col = 'Model name')

        print(f'\n===========Performance metrics for {self.asset_name}===========')
        print(f'Testing period: {self.index[0]} - {self.index[-1]}')
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
