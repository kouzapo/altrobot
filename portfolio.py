#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.stats import norm

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class BacktestPortfolio:
    def __init__(self):
        self.predictions = None

        self.error_metrics = None
        self.performance_metrics = None
    
    def _PT_test(self, predictions, y_true, alpha = 0.05):
        n = len(y_true)
        pyz = accuracy_score(predictions, y_true)

        py = len(y_true[y_true == 1].dropna()) / n
        pz = len(predictions[predictions == 1].dropna()) / n

        p_star = py * pz + (1 - py) * (1 - pz)
        u = p_star * (1 - p_star) / n

        A = (((2 * pz - 1) ** 2) * py * (1 - py)) / n
        B = (((2 * py - 1) ** 2) * pz * (1 - pz)) / n
        C = (4 * py * pz * (1 - py) * (1 - pz)) / (n ** 2)

        w = A + B + C

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
        
        annualized_return = np.power(1 + float(CR), 252 / N) - 1

        self.annualized_return = annualized_return
    
    def _AV(self):
        annualized_volatiliy = float(self.realized_returns.std() * np.sqrt(252))

        self.annualized_volatiliy = annualized_volatiliy
    
    def _SR(self):
        sharpe_ratio = self.annualized_return / self.annualized_volatiliy

        self.sharpe_ratio = sharpe_ratio
    
    def calc_error_metrics(self, predictions, y_true):
        self.accuracy = accuracy_score(predictions, y_true)
        self.precision = precision_score(predictions, y_true)
        self.recall = recall_score(predictions, y_true)
        self.f1 = f1_score(predictions, y_true)
        self.pt_pval = self._PT_test(predictions, y_true)

        self.error_metrics = np.array([self.accuracy, self.precision, self.recall, self.f1, round(self.pt_pval, 6)])
    
    def calc_profitability_metrics(self, signals, returns):
        self._realized_returns(signals, returns)

        self._CR(signals, returns)
        self._AR(len(returns))
        self._AV()
        self._SR()

        self.profitability_metrics = np.array([float(self.cumulative_return.iloc[-1]), self.annualized_return, self.annualized_volatiliy, self.sharpe_ratio])