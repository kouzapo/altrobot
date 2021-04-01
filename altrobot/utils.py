#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json

import pandas as pd
import pandas_datareader.data as pdr
from tensorflow.keras.models import model_from_json, Sequential


KERAS_MODELS_PATH = 'resources/keras_models/'
DATASETS_PATH = 'resources/datasets/'
BANNER_PATH = 'resources/banner'


def show_banner():
    with open(BANNER_PATH, 'r') as f:
        for line in f:
            print(line.splitlines()[0])


def load_model(model_name: str) -> Sequential:
    with open(f'{KERAS_MODELS_PATH + model_name}.json') as f:
        return model_from_json(json.load(f))


def save_model(model: Sequential, name: str):
    if os.path.exists(f'{KERAS_MODELS_PATH + name}.json'):
        raise ValueError(f'Model with name: {name} already exists')

    with open(f'{KERAS_MODELS_PATH + name}.json', 'w') as f:
        json.dump(model.to_json(), f)


def fetch_dataset(symbol: str, start: str, end: str, save: bool = False) -> pd.DataFrame:
    dataset = pdr.DataReader(symbol, 'yahoo', start, end)

    if save:
        dataset.to_csv(f'{DATASETS_PATH + symbol}.dat')

    return dataset.reset_index()


def progress_bar(iteration: int, total: int, prefix: str = '', suffix: str = '', 
                 decimals: int = 2, length: int = 100, fill: str = '█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + ' ' * (length - filled_length)

    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')

    if iteration == total:
        print()
