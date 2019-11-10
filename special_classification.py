import time

import numpy as np
import pandas as pd

from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

import matplotlib.pyplot as plt

'''def data_preprocessing(dataset):
    returns = (dataset['Adj Close'] / dataset['Adj Close'].shift(1) - 1).dropna()
    
    X = pd.DataFrame({
                        'rets -1': returns.shift(1),
                        'rets -2': returns.shift(2),
                        'rets -3': returns.shift(3),
                    })[3:]
    
    y = np.sign(returns)[3:]
    y[y == 0] = 1
    

    return X, y'''

def data_preprocessing(dataset):
    X = dataset.drop(['Volume', 'Date', 'Open', 'High', 'Low', 'Close'], axis = 1)
    X['Adj Close -1'] = X['Adj Close'].shift(1)
    X['Adj Close -2'] = X['Adj Close'].shift(2)
    #X = X.set_index('Date')
    X = X[2:-1]

    y = np.sign((dataset['Adj Close'] - dataset['Adj Close'].shift(1)).dropna())
    y[y == 0] = 1
    y = y.shift(-1).dropna()[1:]
    #y.index = X.index

    returns = (dataset['Adj Close'] / dataset['Adj Close'].shift(1) - 1).dropna()
    returns = returns.shift(-1).dropna()[1:]
    #returns.index = X.index

    return X, y, returns

'''def data_preprocessing(dataset):
    y = np.sign((dataset['Adj Close'] - dataset['Adj Close'].shift(1)).dropna())
    y[y == 0] = 1
    y = y.shift(-1).dropna()

    X = dataset.drop(['Volume', 'Date'], axis = 1)
    X['Adj Close -1'] = X['Adj Close'].shift(1)
    X = X[1:-1]

    returns = (dataset['Adj Close'] / dataset['Adj Close'].shift(1) - 1).dropna()
    returns = returns.shift(-1).dropna()

    #y = dataset['Adj Close'][2:]

    return X, y'''

def run_test(model, param_grid, X_train, y_train, X_test, y_test, ps):
    grid_search = GridSearchCV(model, param_grid, cv = 2, scoring = 'accuracy')
    grid_search.fit(X_train, y_train)

    best = grid_search.best_estimator_
    pred = best.predict(X_test)

    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    print(grid_search.best_estimator_)
    #print('Validation set accuracy: ', grid_search.best_score_)
    print()

    #print(confusion_matrix(y_test, pred))

    return [acc, prec, rec, f1], grid_search.best_estimator_

st = time.perf_counter()

symbols = ['^RUT']#, '^DJI', '^IXIC', '^RUT', 'SPY']

for s in symbols:
    train = pd.read_csv('data_3/' + s + '_train.dat')

    X, y, returns = data_preprocessing(train)

    i = 0
    training_days = 1000
    testing_days = 60

    dataset_length = len(X)
    backtest_periods = []
    accuracy_scores = []

    log_acc = []
    svm_acc = []
    knn_acc = []
    rf_acc = []
    lda_acc = []
    qda_acc = []
    voting_acc = []
    voting2_acc = []

    scaler = StandardScaler()
    #CR = (returns + 1).cumprod() - 1

    


    

    #print(train)
    #print(X)
    #print(y)

    #A = [16.8, 18.12, 18.55, 19.62, 19.22, 19.91, 20.64, 19.21, 20.05, 20.95, 20.49, 20.30, 20.62, 20.77]
    #D = pd.DataFrame([0.05, 0.02, -0.04, -0.1, 0.12, 0.08, -0.03, 0.06, 0.07, -0.02, 0.05])
    D = [0.05, 0.02, -0.04, -0.1, 0.12, 0.08, -0.03, 0.06, 0.07, -0.02, 0.05]
    pred = [1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1]

    cumulative_return = [1]

    A = zip(D, pred)

    for i in A:
        r = i[0]
        p = i[1]

        if p == 1:
            cr = cumulative_return[-1]

            cumulative_return.append(cr * (1 + r))
        
        elif p == -1:
            cr = cumulative_return[-1]

            cumulative_return.append(cr * (1 + 0))
    
    #print(pd.DataFrame(cumulative_return))






    #R = (D / D.shift(1) - 1).dropna()

    #print((D + 1).cumprod() - 1)
    plt.plot(cumulative_return)
    plt.show()



    '''while i + training_days + testing_days <= dataset_length:
        #print('Train start: ' + X[i:i + training_days]['Date'].iloc[0])
        #print('Train end: ' + X[i:i + training_days]['Date'].iloc[-1])
        #print(i, i + training_days)
        #print('Test start: ' + X[i + training_days:i + training_days + testing_days]['Date'].iloc[0])
        #print('Test end: ' + X[i + training_days:i + training_days + testing_days]['Date'].iloc[-1])
        #print(i + training_days, i + training_days + testing_days)
        #print() 
        backtest_periods.append({'Train': [i, i + training_days], 'Test': [i + training_days, i + training_days + testing_days]})
        i += testing_days
    

    
    start = backtest_periods[0]['Test'][0]
    end = backtest_periods[-1]['Test'][1]

    #print(start)
    #print(end)

    print(accuracy_score(y[start:end], np.ones(len(y[start:end]))))
    print(len(backtest_periods))






    

    
    
    
    
    for D in backtest_periods:
        train_i = D['Train']
        test_i = D['Test']

        X_train = X[train_i[0]:train_i[1]]
        y_train = y[train_i[0]:train_i[1]]

        X_test = X[test_i[0]:test_i[1]]
        y_test = y[test_i[0]:test_i[1]]

        test_fold = np.zeros(len(X_train), dtype = np.int)
        test_fold[:800] = -1

        ps = PredefinedSplit(test_fold)



        logreg = LogisticRegression(max_iter = 200)
        logreg_param_grid = [{'C': [6, 7, 8, 9, 10], 'solver': ['liblinear', 'lbfgs']}]

        svm = SVC()
        svm_param_grid = [{'C': [6, 7, 8, 9, 10], 'gamma': ['auto', 'scale']}]

        knn = KNeighborsClassifier()
        knn_param_grid = [{'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8]}]

        rf = RandomForestClassifier()
        rf_param_grid = [{'n_estimators': [10, 30, 50, 70, 100], 'max_depth': [10, 20, 30, 40, 50], 'criterion': ['gini', 'entropy']}]

        lda = LinearDiscriminantAnalysis()
        lda_param_grid = [{'solver': ['svd', 'lsqr', 'eigen']}]

        qda = QuadraticDiscriminantAnalysis()
        qda_param_grid = [{'tol': [1.0e-4]}]



        logreg_res, log_best = run_test(logreg, logreg_param_grid, scaler.fit_transform(X_train), y_train, scaler.fit_transform(X_test), y_test, ps)
        svm_res, svm_best = run_test(svm, svm_param_grid, scaler.fit_transform(X_train), y_train, scaler.fit_transform(X_test), y_test, ps)
        knn_res, knn_best = run_test(knn, knn_param_grid, X_train, y_train, X_test, y_test, ps)
        lda_res, lda_best = run_test(lda, lda_param_grid, X_train, y_train, X_test, y_test, ps)
        qda_res, qda_best = run_test(qda, qda_param_grid, X_train, y_train, X_test, y_test, ps)
        rf_res, rf_best = run_test(rf, rf_param_grid, X_train, y_train, X_test, y_test, ps)

        #--------------------------------
        voting_clf = VotingClassifier(estimators = [('lr', log_best), ('svm', svm_best), ('lda', lda_best)], voting = 'hard')
        voting_clf.fit(scaler.fit_transform(X_train), y_train)

        pred = voting_clf.predict(scaler.fit_transform(X_test))

        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred)
        rec = recall_score(y_test, pred)
        f1 = f1_score(y_test, pred)

        voting_res = [acc, prec, rec, f1]


        #--------------------------------
        voting_clf2 = VotingClassifier(estimators = [('rf', rf_best), ('lda', lda_best), ('qda', qda_best), ('knn', knn_best)], voting = 'hard')
        voting_clf2.fit(X_train, y_train)

        pred = voting_clf2.predict(X_test)

        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred)
        rec = recall_score(y_test, pred)
        f1 = f1_score(y_test, pred)

        voting_res2 = [acc, prec, rec, f1]


        results = pd.DataFrame(np.array([logreg_res, svm_res, knn_res, rf_res, lda_res, qda_res, voting_res, voting_res2]))
        results.index = ['Logistic Regression', 'Support Vector Machine', 'KNN', 'Random Forest', 'LDA', 'QDA', 'Voting', 'Voting_2']
        results.columns = ['Accuracy', 'Precision', 'Recall', 'F1']
        up = accuracy_score(y_test, np.ones(len(y_test)))

        accuracy_scores.append(results['Accuracy'].max())

        log_acc.append(results['Accuracy'].iloc[0])
        svm_acc.append(results['Accuracy'].iloc[1])
        knn_acc.append(results['Accuracy'].iloc[2])
        rf_acc.append(results['Accuracy'].iloc[3])
        lda_acc.append(results['Accuracy'].iloc[4])
        qda_acc.append(results['Accuracy'].iloc[5])

        voting_acc.append(results['Accuracy'].iloc[6])
        voting2_acc.append(results['Accuracy'].iloc[7])

        print('Always up accuracy:', up)
        print('------------------------Results for {}----------------------'.format(s))
        print(results)
        print()
    
    print('Total accuracy:', sum(accuracy_scores) / len(accuracy_scores))

    print('Logistic Regression accuracy:', sum(log_acc) / len(log_acc))
    print('Support Vector Machine accuracy:', sum(svm_acc) / len(svm_acc))
    print('KNN accuracy:', sum(knn_acc) / len(knn_acc))
    print('Random Forest accuracy:', sum(rf_acc) / len(rf_acc))
    print('LDA accuracy:', sum(lda_acc) / len(lda_acc))
    print('QDA accuracy:', sum(qda_acc) / len(qda_acc))

    print('Voting accuracy:', sum(voting_acc) / len(voting_acc))
    print('Voting 2 accuracy:', sum(voting2_acc) / len(voting2_acc))'''



print('\nExecution time: {}'.format(time.perf_counter() - st))