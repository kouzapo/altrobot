#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List

import pandas as pd
from sklearn.preprocessing import StandardScaler


def generate_subsets(X: pd.DataFrame, y: pd.Series, returns: pd.Series, training_size: int, window: int) -> List[dict]:
	backtest_subsets = []
	n = len(X)
	i = 0

	while i + training_size + window <= n:
		subset = {}

		subset['X_train'] = X[i:i + training_size]
		subset['X_test'] = X[i + training_size:i + training_size + window]

		subset['y_train'] = y[i:i + training_size]
		subset['y_test'] = y[i + training_size:i + training_size + window]

		subset['returns_train'] = returns[i:i + training_size]
		subset['returns_test'] = returns[i + training_size:i + training_size + window]

		backtest_subsets.append(subset)

		i += window
	
	if len(y) % window != 0:
		i -= window

		backtest_subsets[-1]['X_test'] = X[i + training_size:len(X)]
		backtest_subsets[-1]['y_test'] = y[i + training_size:len(y)]
		backtest_subsets[-1]['returns_test'] = returns[i + training_size:len(returns)]

	return backtest_subsets


def standardize(backtest_subsets: List[dict]) -> List[dict]:
	scaler = StandardScaler()
	standardized_subsets = []

	for subset in backtest_subsets:
		standardized_subsets.append({'X_train': scaler.fit_transform(subset['X_train']),
									'X_test': scaler.fit_transform(subset['X_test']),
									'y_train': subset['y_train'],
									'y_test': subset['y_test'],
									'returns_train': subset['returns_train'],
									'returns_test': subset['returns_test']})

	return standardized_subsets
