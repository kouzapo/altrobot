#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Portfolio:
    def __init__(self, cash):
        self.cash = cash

        self.predictions = None

        self.error_metrics = None
        self.performance_metrics = None
    
    def calcErrorMetrics(self, predictions, y_true):
        accuracy = accuracy_score(predictions, y_true)
        precision = precision_score(predictions, y_true)
        recall = recall_score(predictions, y_true)
        f1 = f1_score(predictions, y_true)

        self.error_metrics = np.array([accuracy, precision, recall, f1])
    
    def calcProfitabilityMetrics(self, signals):
        pass