#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def generate_subsets(
		X: pd.DataFrame,
		y: pd.Series,
		returns: pd.Series,
		training_size: int,
		window: int
	) -> List[dict]:
	backtest_subsets = []
	n = len(X)
	i = 0

	while i + training_size + window <= n:
		backtest_subsets.append({
			'X_train': X[i:i + training_size],
			'X_test': X[i + training_size:i + training_size + window],
			'y_train': y[i:i + training_size],
			'y_test': y[i + training_size:i + training_size + window],
			'returns_train': returns[i:i + training_size],
			'returns_test': returns[i + training_size:i + training_size + window]
		})

		i += window
	
	if len(y) % window != 0:
		i -= window

		backtest_subsets[-1]['X_test'] = X[i + training_size:len(X)]
		backtest_subsets[-1]['y_test'] = y[i + training_size:len(y)]
		backtest_subsets[-1]['returns_test'] = returns[i + training_size:len(returns)]

	return backtest_subsets

def generate_LSTM_subsets(
		X: pd.DataFrame,
		y: pd.Series,
		returns: pd.Series,
		training_size: int,
		timesteps: int,
		window: int
	) -> List[dict]:
	backtest_subsets = []
	n = len(X)
	i = timesteps - 1

	while i + training_size + window <= n:
		backtest_subsets.append({
			'X_train': np.array([X[i - timesteps + j:i + j] for j in range(1, training_size + 1)]),
			'X_test': np.array([X[i + training_size - timesteps + j:i + training_size + j] for j in range(1, window + 1)]),
			'y_train': y[i:i + training_size],
			'y_test': y[i + training_size:i + training_size + window],
			'returns_train': returns[i:i + training_size],
			'returns_test': returns[i + training_size:i + training_size + window]
		})

		i += window
	
	return backtest_subsets

def standardize(backtest_subsets: List[dict]) -> List[dict]:
	scaler = StandardScaler()

	for subset in backtest_subsets:
		subset['X_train'] = scaler.fit_transform(subset['X_train'])
		subset['X_test'] = scaler.fit_transform(subset['X_test'])

	return backtest_subsets

def standardize_LSTM(backtest_subsets: List[dict]) -> List[dict]:
	scaler = StandardScaler()

	for subset in backtest_subsets:
		subset['X_train'] = np.array([scaler.fit_transform(timeframe) for timeframe in subset['X_train']])
		subset['X_test'] = np.array([scaler.fit_transform(timeframe) for timeframe in subset['X_test']])
	
	return backtest_subsets
