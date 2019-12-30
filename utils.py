#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json

import pandas as pd
import pandas_datareader.data as pdr

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

from keras.models import model_from_json

sys.stderr = stderr

def banner():
    with open('banner.txt', 'r') as f:
        for i in f:
            print(i.splitlines()[0])

def load_model(model_name):
    with open(model_name + '.json') as f:
        return model_from_json(json.load(f))

def progress_bar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '#'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    #bar = fill * filled_length + '-' * (length - filled_length)
    bar = fill * filled_length + ' ' * (length - filled_length)
    #█
    
    #print('\r%s %s%% %s' % (prefix, percent, suffix), end = '\r') #2
    #print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r') #1
    print('\r%s [%s] %s%% %s' % (prefix, bar, percent, suffix), end = '\r') #3
    
    if iteration == total:
        print()

def get_data(symbols, start, end):
    for s in symbols:
        D = pdr.DataReader(s, 'yahoo', start, end)

        D.to_csv(s + '.dat')

        print(s)
