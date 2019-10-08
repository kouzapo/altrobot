import time

import numpy as np
import pandas as pd

from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
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
    #y.index = X.index
    y.name = 'Sign'

    returns = (dataset['Adj Close'] / dataset['Adj Close'].shift(1) - 1).dropna()
    returns = returns.shift(-1).dropna()[1:]
    returns.index = X.index
    returns.name = 'Returns'

    return X, y, returns

st = time.perf_counter()



index_name = '^IXIC'

train = pd.read_csv('data_3/' + index_name + '_train.dat')
X, y, returns = data_preprocessing(train)

#print(X)
#print(y)
#print(returns)




svm_param_grid = [{'C': [5, 6, 7, 8, 9, 10, 11, 12], 'gamma': ['auto', 'scale']}]
logreg_param_grid = [{'C': [5, 6, 7, 8, 9, 10, 11, 12], 'solver': ['liblinear', 'lbfgs']}]
dt_param_grid = [{'max_depth': [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65], 'criterion': ['gini', 'entropy']}]
lda_param_grid = [{'solver': ['svd', 'lsqr', 'eigen']}]

b = Backtester(X, y, returns, index_name)
b.splitTrainTest(rolling_window_split = (1000, 20))
#b.splitTrainTest(single_split = 2000) 





svm = Model(SVC(), svm_param_grid, 'Support Vector Machine', scaling = True)
log = Model(LogisticRegression(max_iter = 200), logreg_param_grid, 'Logistic Regression', scaling = True)
dt = Model(DecisionTreeClassifier(), dt_param_grid, 'Decision Tree')
lda = Model(LinearDiscriminantAnalysis(), lda_param_grid, 'LDA')











b.addModels([svm, log, dt, lda])
b.testModels(['Support Vector Machine', 'Logistic Regression', 'Decision Tree', 'LDA'])
b.calcPerformanceMetrics(['Support Vector Machine', 'Logistic Regression', 'Decision Tree', 'LDA'])



print('\nExecution time: {}'.format(time.perf_counter() - st))