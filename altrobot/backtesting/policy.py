#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import pandas as pd


class Policy(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def generate_signals(self, predicted_probs): ...


class AllInOutPolicy(Policy):

    def __init__(
            self,
            bounds: Tuple[float, float]
        ):
        if bounds[0] > bounds[1]:
             raise ValueError('Lower bound is greater than the upper bound')

        self.bounds = bounds

    def generate_signals(
            self,
            predicted_probs: pd.Series
        ) -> np.ndarray:
        lower_bound = self.bounds[0]
        upper_bound = self.bounds[1]

        signals = []

        for p in predicted_probs:
            if p >= upper_bound:
                signals.append(1)
            elif p < lower_bound:
                signals.append(0)
            else:
                if len(signals) == 0:
                    signals.append(0)
                else:
                    signals.append(signals[-1])
        
        return np.array(signals)
