#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.stats import norm

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

class BacktestPortfolio:
    def __init__(self):
        self.error_metrics = []
        self.performance_metrics = []
    
    def _PT_test(self, predictions, y_true):
        n = len(y_true)
        pyz = accuracy_score(predictions, y_true)

        py = len(y_true[y_true == 1].dropna()) / n
        pz = len(predictions[predictions == 1].dropna()) / n

        p_star = py * pz + (1 - py) * (1 - pz)
        u = p_star * (1 - p_star) / n

        w = (((2 * pz - 1) ** 2) * py * (1 - py)) / n + \
            (((2 * py - 1) ** 2) * pz * (1 - pz)) / n + \
            (4 * py * pz * (1 - py) * (1 - pz)) / (n ** 2)

        PT = (pyz - p_star) / np.sqrt(u - w)
        p_value = 1 - norm.cdf(PT)

        return p_value
    
    def _realized_returns(self, signals, returns):
        self.realized_returns = np.multiply(signals, np.array(returns))
    
    def _CR(self, signals, returns):
        cumulative_return = [1]

        for i in zip(signals, returns):
            s = i[0]
            r = i[1]

            cr = cumulative_return[-1]

            cumulative_return.append(cr + (cr * s * r))
        
        self.cumulative_return = pd.Series(cumulative_return) - 1
    
    def _AR(self, N):
        CR = self.cumulative_return.iloc[-1]

        self.annualized_return = np.power(1 + float(CR), 252 / N) - 1
    
    def _AV(self):
        self.annualized_volatiliy = float(self.realized_returns.std() * np.sqrt(252))
    
    def _SR(self):
        self.sharpe_ratio = self.annualized_return / self.annualized_volatiliy

    def _IR(self, returns, bnh_AR):
        traking_error = float((self.realized_returns - returns).std() * np.sqrt(252))

        self.information_ratio = (self.annualized_return - bnh_AR) / traking_error
    
    def calc_error_metrics(self, predictions, y_true):
        self.accuracy = accuracy_score(predictions, y_true)
        self.precision = recall_score(predictions, y_true)
        self.recall = precision_score(predictions, y_true)
        self.f1 = f1_score(predictions, y_true)
        self.pt_pval = self._PT_test(predictions, y_true)

        self.error_metrics = np.array([self.accuracy, self.precision, self.recall, self.f1, round(self.pt_pval, 6)])
    
    def calc_profitability_metrics(self, signals, returns, *bnh_AR):
        self._realized_returns(signals, returns)

        self._CR(signals, returns)
        self._AR(len(returns))
        self._AV()
        self._SR()
        
        if bnh_AR:
            self._IR(returns, bnh_AR[0])
        else:
            self.information_ratio = 0

        self.profitability_metrics = np.array([float(self.cumulative_return.iloc[-1]),
                                                     self.annualized_return,
                                                     self.annualized_volatiliy,
                                                     self.sharpe_ratio,
                                                     self.information_ratio])
    
    def calc_conf_matrix(self, predictions, y_true):
        conf_matrix = confusion_matrix(predictions, y_true)
        
        self.conf_matrix = np.array([conf_matrix[1][1], conf_matrix[0][0], conf_matrix[1][0], conf_matrix[0][1]])

    def calc_conf_matrix_prof(self, predictions, y_true, returns):
        A = pd.DataFrame({'y_true': y_true, 'pred': predictions, 'rets': returns})

        TP = A[(A['y_true'] == 1) & (A['pred'] == 1)]
        TN = A[(A['y_true'] == 0) & (A['pred'] == 0)]
        FP = A[(A['y_true'] == 0) & (A['pred'] == 1)]
        FN = A[(A['y_true'] == 1) & (A['pred'] == 0)]

        self.conf_matrix_prof = np.array([TP['rets'].mean(), TN['rets'].mean(), FP['rets'].mean(), FN['rets'].mean()])