#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

import numpy as np

class Policy(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def generate_signals(self):
        pass

class AllInOutPolicy(Policy):
    def __init__(self):
        pass
    
    def generate_signals(self, predictions, predicted_probs):

        return np.array([1 if p == 1 else 0 for p in predictions])

class BuyHoldSalePolicy(Policy):
    def __init__(self):
        pass

    def generate_signals(self, predictions, predicted_probs):
        signals = []

        for p in predicted_probs:
            if p >= 0.6:
                signals.append(1)
            elif p <= 0.4:
                signals.append(0)
            else:
                if len(signals) == 0:
                    signals.append(0)
                else:
                    signals.append(signals[-1])

        return np.array(signals) 