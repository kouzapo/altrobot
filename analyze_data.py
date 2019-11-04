#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time

import numpy as np
import pandas as pd

from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

from model import Model
from strategy import AllInOutStrategy
from portfolio import Portfolio
from backtesting import Backtester
from data_preprocessing import threePastClosing

st = time.perf_counter()

index_name = '^GSPC'


train = pd.read_csv('data_3/' + index_name + '_train.dat')
X, y, returns = threePastClosing(train)

logreg_param_grid = [{'C': [5, 6, 7, 8, 9, 10, 11, 12], 'solver': ['liblinear', 'lbfgs']}]
log_model = Model(LogisticRegression(max_iter = 200), logreg_param_grid, 'Logistic Regression', scaling = True)

strategy = AllInOutStrategy()
portfolio = Portfolio()

b = Backtester(X, y, returns, log_model, strategy, portfolio)

b.splitTrainTest(by_index = {'Train': (0, 1000), 'Test':(1000, 3520)}, single_split = False, window = 10)
#b.splitTrainTest(by_index = {'Train': (0, 2000), 'Test':(2000, 3520)}, single_split = True)
#b.splitTrainTest(by_date = {'Train': ('2005-01-05', '2012-12-13'), 'Test':('2012-12-14', '2018-12-28')}, single_split = True)


'''for i in b.backtest_periods:
    print(i)'''

b.runTest()



print('\nExecution time: {}'.format(time.perf_counter() - st))