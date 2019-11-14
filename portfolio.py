#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Portfolio:
    def __init__(self):
        self.predictions = None

        self.error_metrics = None
        self.performance_metrics = None
    
    def __calc_daily_returns(self, signals, returns):
        self.daily_returns = np.multiply(np.array(signals), np.array(returns))
    
    def __calc_CR(self, signals, returns):
        cumulative_return = [1]

        A = zip(signals, returns)

        for i in A:
            s = i[0]
            r = i[1]

            cr = cumulative_return[-1]

            cumulative_return.append(cr + (cr * s * r))
        
        self.cumulative_return = pd.DataFrame(cumulative_return) - 1
    
    def __calc_AR(self, N):
        CR = self.cumulative_return.iloc[-1]
        
        annualized_return = np.power(1 + float(CR), 252 / N) - 1

        self.annualized_return = annualized_return
    
    def __calc_AV(self):
        annualized_volatiliy = float(self.daily_returns.std() * np.sqrt(252))

        self.annualized_volatiliy = annualized_volatiliy
    
    def __calc_SR(self):
        sharpe_ratio = self.annualized_return / self.annualized_volatiliy

        self.sharpe_ratio = sharpe_ratio
    
    def calc_error_metrics(self, predictions, y_true):
        self.accuracy = accuracy_score(predictions, y_true)
        self.precision = precision_score(predictions, y_true)
        self.recall = recall_score(predictions, y_true)
        self.f1 = f1_score(predictions, y_true)

        self.error_metrics = np.array([self.accuracy, self.precision, self.recall, self.f1])
    
    def calc_profitability_metrics(self, signals, returns):
        self.__calc_daily_returns(signals, returns)

        self.__calc_CR(signals, returns)
        self.__calc_AR(len(returns))
        self.__calc_AV()
        self.__calc_SR()

        self.profitability_metrics = np.array([float(self.cumulative_return.iloc[-1]), self.annualized_return, self.annualized_volatiliy, self.sharpe_ratio])