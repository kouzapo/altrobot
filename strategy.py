#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

import numpy as np

class Strategy(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def generate_signals(self):
        pass

class AllInOutStrategy(Strategy):
    def __init__(self):
        pass
    
    def generate_signals(self, predictions):
        signals = []

        for p in predictions:
            if p >= 0.5:
                signals.append(1)

            elif p < 0.5:
                signals.append(0)
        
        return np.array(signals)