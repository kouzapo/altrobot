import itertools

import numpy as np
import pandas as pd

from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import style

style.use('ggplot')

class Backtester:
    def __init__(self, X, y, returns, asset_name):
        self.X = X
        self.y = y
        self.returns = returns
        self.asset_name = asset_name

        self.models = {}
        self.predictions = {}
        self.cumulative_returns = {}
        self.backtest_periods = []
    
    def __runGridSearch(self, model, X_train, y_train, X_test, y_test):
        grid_search = GridSearchCV(model.estimator, model.hyperparams, cv = 2, scoring = 'accuracy')
        grid_search.fit(X_train, y_train)

        best_estimator = grid_search.best_estimator_
        predictions = best_estimator.predict(X_test)
        model.estimator = best_estimator

        return predictions
    
    def __calcCR(self, returns, predictions):
        cumulative_return = [1]

        A = zip(returns, predictions)

        for i in A:
            r = i[0]
            p = i[1]

            if p == 1:
                cr = cumulative_return[-1]

                cumulative_return.append(cr * (1 + r))
            
            elif p == -1:
                cr = cumulative_return[-1]
                
                cumulative_return.append(cr * (1 + 0))

        return pd.DataFrame(cumulative_return) - 1
    
    def __calcErrorMetrics(self, model_names):
        backtest_periods = self.backtest_periods
        y = self.y

        start = backtest_periods[0]['Test'][0]
        end = backtest_periods[-1]['Test'][1]

        results = []

        for name in model_names:
            predictions = self.predictions[name]

            accuracy = accuracy_score(predictions, y[start:end])
            precision = precision_score(predictions, y[start:end])
            recall = recall_score(predictions, y[start:end])
            f1 = f1_score(predictions, y[start:end])

            results.append([accuracy, precision, recall, f1])
        
        report = pd.DataFrame(np.array(results))
        report.index = model_names
        report.columns = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

        #print(report)
        return report
    
    def __calcProfitabilityMetrics(self, model_names):
        backtest_periods = self.backtest_periods
        predictions = self.predictions
        returns = self.returns

        start = backtest_periods[0]['Test'][0]
        end = backtest_periods[-1]['Test'][1]

        results = []

        for name in model_names:
            cumulative_return = self.__calcCR(returns[start:end], predictions[name])

            self.cumulative_returns[name] = cumulative_return
            results.append([float(cumulative_return.iloc[-1])])
        
        report = pd.DataFrame(results)
        report.index = model_names
        report.columns = ['Cumulative Return']

        #print(report)
        return report
    
    def __calcBenchmarkMetrics(self):
        backtest_periods = self.backtest_periods
        returns = self.returns
        y = self.y

        results = []
        
        start = backtest_periods[0]['Test'][0]
        end = backtest_periods[-1]['Test'][1]

        BnH_pred = np.ones(len(y[start:end]))

        perfect_pred = np.sign(returns[start:end])
        perfect_pred[perfect_pred == 0] = 1

        #-----BnH-----
        accuracy = accuracy_score(y[start:end], BnH_pred)
        precision = precision_score(y[start:end], BnH_pred)
        recall = recall_score(y[start:end], BnH_pred)
        f1 = f1_score(y[start:end], BnH_pred)
        cumulative_return = (returns[start:end] + 1).cumprod() - 1

        self.cumulative_returns['BnH'] = cumulative_return
        results.append([accuracy, precision, recall, f1, cumulative_return.iloc[-1]])

        '''#-----Perfect-----
        accuracy = accuracy_score(y[start:end], perfect_pred)
        precision = precision_score(y[start:end], perfect_pred)
        recall = recall_score(y[start:end], perfect_pred)
        f1 = f1_score(y[start:end], perfect_pred)
        cumulative_return = pd.DataFrame(self.__calcCR(returns[start:end], perfect_pred)) - 1

        results.append([accuracy, precision, recall, f1, cumulative_return.iloc[-1]])'''
        
        
        report = pd.DataFrame(results)
        report.index = ['Buy and Hold']#, 'Perfect Predictions']
        report.columns = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Cumulative Return']

        #print(report)

        return report
    
    def __plotCR(self, names):
        cumulative_returns = self.cumulative_returns

        for name in names:
            cumulative_return = cumulative_returns[name]

            plt.plot(np.array(cumulative_return), label = name)

            plt.xlabel('Time')
            plt.ylabel('Cumulative Return')
            plt.legend(loc = 2)
            plt.title('Cumulative return for {}'.format(self.asset_name))

        plt.show()


        '''plt.plot(np.array(cumulative_return), color = 'blue', label = name)

        plt.xlabel('Days')
        plt.ylabel('Cumulative Return')
        plt.legend(loc = 2)
        plt.title('Cumulative return for {}'.format(name))

        plt.show()'''
    
    def addModels(self, models):
        for model in models:
            self.models[model.name] = model
    
    def splitTrainTest(self, single_split = None, rolling_window_split = None):
        i = 0
        dataset_length = len(self.X)

        if single_split is not None:
            self.backtest_periods.append({'Train': (0, single_split), 'Test': (single_split, dataset_length)})
        
        if rolling_window_split is not None:
            training_days = rolling_window_split[0]
            testing_days = rolling_window_split[1]

            while i + training_days + testing_days <= dataset_length:
                self.backtest_periods.append({'Train': (i, i + training_days), 'Test': (i + training_days, i + training_days + testing_days)})

                i += testing_days
    
    def testModels(self, model_names, scaling = False):
        backtest_periods = self.backtest_periods
        X = self.X
        y = self.y

        scaler = StandardScaler()

        for name in model_names:
            model = self.models[name]

            self.predictions[name] = []

        for P in backtest_periods:
            train_i = P['Train']
            test_i = P['Test']

            X_train = X[train_i[0]:train_i[1]]
            y_train = y[train_i[0]:train_i[1]]

            X_test = X[test_i[0]:test_i[1]]
            y_test = y[test_i[0]:test_i[1]]

            for name in model_names:
                model = self.models[name]

                if model.scaling:
                    predictions = self.__runGridSearch(model, scaler.fit_transform(X_train), y_train, scaler.fit_transform(X_test), y_test)
                    self.predictions[name].append(predictions)
                else:
                    predictions = self.__runGridSearch(model, X_train, y_train, X_test, y_test)
                    self.predictions[name].append(predictions)

            print(P)
        
        for name in model_names:
             total_predictions = list(itertools.chain.from_iterable(self.predictions[name]))
             self.predictions[name] = total_predictions
    
    def calcPerformanceMetrics(self, model_names):
        backtest_periods = self.backtest_periods
        X = self.X

        start = backtest_periods[0]['Test'][0]
        end = backtest_periods[-1]['Test'][1]

        benchmark_report = self.__calcBenchmarkMetrics()
        accuracy_report = self.__calcErrorMetrics(model_names)
        profitability_report = self.__calcProfitabilityMetrics(model_names)

        print()
        print('Performance metrics for:', self.asset_name)
        print('Testing period: {} to {}'.format(X.index[start], X.index[end - 1]))
        print('Number of models tested:', len(model_names))
        print()

        print('---------------------------Benchmark Metrics--------------------------')
        print(benchmark_report)
        print()

        print('------------------------Prediction Error-----------------------')
        print(accuracy_report)
        print()

        print('--------------Profitability--------------')
        print(profitability_report)

        self.__plotCR(['BnH', 'Logistic Regression'])