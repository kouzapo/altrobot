#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import pandas_datareader.data as pdr

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
