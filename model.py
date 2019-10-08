import numpy as np
import pandas as pd

class Model:
    def __init__(self, estimator, hyperparams, name, scaling = False):
        self.estimator = estimator
        self.hyperparams = hyperparams
        self.name = name
        self.scaling = scaling