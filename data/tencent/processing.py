#!/usr/bin/env python
# coding=utf-8


# processing dataset using pandas

import pandas as pd
import numpy as np
data_file = 'output_mz'
data = pd.read_csv(data_file,header = None,sep = '\t',names = ['s1','s2','flag'])
print('data_positive_label"{}'.format(data['flag'].mean()))
print('all_sample_length:{}'.format(len(data)))
print('total number of query: {}'.format(len(data['s1'].unique())))
print(data[data['flag']==1])

msk = np.random.rand(len(data)) < 0.8

train = data[msk]

test = data[~msk]
print(len(train))
print(len(test))

train.to_csv('train.txt',header = None,index = None,sep = '\t')
test.to_csv('test.txt',header = None,index = None,sep ='\t')
test.to_csv('dev.txt',header = None, index = None, sep = '\t')

