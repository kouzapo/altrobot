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
    
    def __calcDailyReturns(self, signals, returns):
        daily_returns = [0]

        A = zip(signals, returns)

        for i in A:
            s = i[0]
            r = i[1]

            if s[0] == 'buy':
                daily_returns.append(r)
            
            elif s[0] == 'sell':
                daily_returns.append(0)
        
        self.daily_returns = pd.DataFrame(daily_returns)
    
    def __calcCR(self, signals, returns):
        cumulative_return = [1]

        A = zip(signals, returns)

        for i in A:
            s = i[0]
            r = i[1]

            cr = cumulative_return[-1]

            if s[0] == 'buy':
                cumulative_return.append(cr * (1 + r))
            
            elif s[0] == 'sell':
                cumulative_return.append(cr * (1 + 0))
        
        self.cumulative_return = pd.DataFrame(cumulative_return) - 1
    
    def __calcAR(self, N):
        CR = self.cumulative_return.iloc[-1]
        
        annualized_return = np.power(1 + float(CR), 252 / N) - 1

        self.annualized_return = annualized_return
    
    def __calcAV(self):
        annualized_volatiliy = float(self.daily_returns.std() * np.sqrt(252))

        self.annualized_volatiliy = annualized_volatiliy
    
    def __calcSR(self):
        sharpe_ratio = self.annualized_return / self.annualized_volatiliy

        self.sharpe_ratio = sharpe_ratio
    
    def calcErrorMetrics(self, predictions, y_true):
        self.accuracy = accuracy_score(predictions, y_true)
        self.precision = precision_score(predictions, y_true)
        self.recall = recall_score(predictions, y_true)
        self.f1 = f1_score(predictions, y_true)

        self.error_metrics = np.array([self.accuracy, self.precision, self.recall, self.f1])
    
    def calcProfitabilityMetrics(self, signals, returns):
        self.__calcDailyReturns(signals, returns)

        self.__calcCR(signals, returns)
        self.__calcAR(len(returns))
        self.__calcAV()
        self.__calcSR()

        self.profitability_metrics = np.array([float(self.cumulative_return.iloc[-1]), self.annualized_return, self.annualized_volatiliy, self.sharpe_ratio])