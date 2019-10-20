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
from backtesting import Backtester

def data_preprocessing(dataset):
    #dates = dataset['Date']

    X = dataset.drop(['Volume', 'Open', 'High', 'Low', 'Close'], axis = 1)
    X['Adj Close -1'] = X['Adj Close'].shift(1)
    X['Adj Close -2'] = X['Adj Close'].shift(2)
    #X = X.set_index('Date')
    X = X[2:-1]

    dates = X['Date']

    X = X.drop(['Date'], axis = 1)
    X.index = dates

    y = np.sign((dataset['Adj Close'] - dataset['Adj Close'].shift(1)).dropna())
    y[y == 0] = 1
    y = y.shift(-1).dropna()[1:]
    y.index = X.index
    y.name = 'Sign'

    returns = (dataset['Adj Close'] / dataset['Adj Close'].shift(1) - 1).dropna()
    returns = returns.shift(-1).dropna()[1:]
    returns.index = X.index
    returns.name = 'Returns'

    return X, y, returns

st = time.perf_counter()

index_name = '^GSPC'


train = pd.read_csv('data_3/' + index_name + '_train.dat')
X, y, returns = data_preprocessing(train)







svm_param_grid = [{'C': [5, 6, 7, 8, 9, 10, 11, 12], 'gamma': ['auto', 'scale']}]
logreg_param_grid = [{'C': [5, 6, 7, 8, 9, 10, 11, 12], 'solver': ['liblinear', 'lbfgs']}]
dt_param_grid = [{'max_depth': [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65], 'criterion': ['gini', 'entropy']}]
lda_param_grid = [{'solver': ['svd', 'lsqr', 'eigen']}]

b = Backtester(X, y, returns, index_name)

b.splitTrainTest(by_index = {'Train': (0, 1000), 'Test':(1000, 3520)}, single_split = False, window = 10)
#b.splitTrainTest(by_index = {'Train': (0, 2000), 'Test':(2000, 3520)}, single_split = True)
#b.splitTrainTest(by_date = {'Train': ('2005-01-05', '2012-12-13'), 'Test':('2012-12-14', '2018-12-28')}, single_split = True)

#b.splitTrainTest(by_date = {'Train': ('2005-01-05', '2010-09-30'), 'Test':('2010-10-01', '2018-05-01')}, single_split = True) 
#b.splitTrainTest(by_date = {'Train': ('2005-01-05', '2010-09-30'), 'Test':('2010-10-01', '2018-05-01')}, single_split = False, window = 10) 

#b.backtest_periods[-1]['Test'] = (3325, 3353) #20
#b.backtest_periods[-1]['Test'] = (3335, 3353) #10

'''for i in b.backtest_periods:
    print(i)'''




svm = Model(SVC(), svm_param_grid, 'Support Vector Machine', scaling = True)
log = Model(LogisticRegression(max_iter = 200), logreg_param_grid, 'Logistic Regression', scaling = True)
dt = Model(DecisionTreeClassifier(), dt_param_grid, 'Decision Tree')
lda = Model(LinearDiscriminantAnalysis(), lda_param_grid, 'LDA')








b.addModels([svm, log, dt, lda])
b.testModels(['Support Vector Machine', 'Logistic Regression', 'Decision Tree', 'LDA'])
b.calcPerformanceMetrics(['Support Vector Machine', 'Logistic Regression', 'Decision Tree', 'LDA'])

print('\nExecution time: {}'.format(time.perf_counter() - st))
