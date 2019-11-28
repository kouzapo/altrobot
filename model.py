#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler

class Model:
    def __init__(self, estimator, hyperparams, name, scaling = False):
        self.estimator = estimator
        self.hyperparams = hyperparams
        self.name = name
        self.scaling = scaling

        self.best_estimator = None

    def grid_search(self, X_train, y_train):
        if self.scaling:
            X_train = StandardScaler().fit_transform(X_train)

        test_fold = np.zeros(len(X_train), dtype = np.int)
        test_fold[:950] = -1

        ps = PredefinedSplit(test_fold)

        grid_search = GridSearchCV(self.estimator, self.hyperparams, cv = ps, scoring = 'accuracy')
        grid_search.fit(X_train, y_train)

        self.best_estimator = grid_search.best_estimator_
    
    def predict(self, X_test):
        if self.scaling:
            X_test = StandardScaler().fit_transform(X_test)

        predictions = self.best_estimator.predict(X_test)
        #predictions = self.best_estimator.predict_proba(X_test)[:, 1]

        return predictions