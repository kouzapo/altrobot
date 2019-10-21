#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

class Model:
    def __init__(self, estimator, hyperparams, name, scaling = False):
        self.estimator = estimator
        self.hyperparams = hyperparams
        self.name = name
        self.scaling = scaling

        self.predictions = []
        self.daily_returns = []
        self.cumulative_returns = []
    
    def calcDailyReturns(self, returns, predictions):
        daily_returns = [0]

        A = zip(returns, predictions)

        for i in A:
            r = i[0]
            p = i[1]

            if p == 1:
                daily_returns.append(r)
            
            elif p == -1:
                daily_returns.append(0)
        
        self.daily_returns = pd.DataFrame(daily_returns)
        
        return pd.DataFrame(daily_returns)
    
    def calcCR(self, returns):
        cumulative_returns = [1]

        A = zip(returns, self.predictions)

        for i in A:
            r = i[0]
            p = i[1]

            if p == 1:
                cr = cumulative_returns[-1]

                cumulative_returns.append(cr * (1 + r))
            
            elif p == -1:
                cr = cumulative_returns[-1]
                
                cumulative_returns.append(cr)
        
        self.cumulative_returns = pd.DataFrame(cumulative_returns) - 1
        
        return self.cumulative_returns
    
    def calcAR(self, N):
        annualized_return = np.power(1 + float(self.cumulative_returns.iloc[-1]), 252 / N) - 1

        self.annualized_return = annualized_return

        return annualized_return
    
    def calcAV(self, returns):
        daily_returns = self.calcDailyReturns(returns, self.predictions)

        annualized_volatiliy = float(daily_returns.std() * np.sqrt(252))

        self.annualized_volatiliy = annualized_volatiliy

        return annualized_volatiliy
    
    def calcSharpeRatio(self):
        sharpe_ratio = self.annualized_return / self.annualized_volatiliy

        return sharpe_ratio
    
    def calcDD(self):
        pass