#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

class Strategy(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def generateSignals(self):
        pass

class AllInOutStrategy(Strategy):
    def __init__(self):
        pass
    
    def generateSignals(self, predictions):
        signals = []

        for p in predictions:
            if p == 1:
                signals.append(('buy', 1))

            elif p == -1:
                signals.append(('sell', 1))