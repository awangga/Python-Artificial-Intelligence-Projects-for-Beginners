# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 13:19:52 2021

@author: Dell
"""


import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def prepoc(datapath):
    d = pd.read_csv(datapath, sep='\s+', header=None, error_bad_lines=False, warn_bad_lines=False, names=['index',	'mode',	'ttme',	'invc',	'invt',	'gc',	'hinc',	'psize'])

    d_train_att = d.iloc[:, :7]
    d_train_pass = d.iloc[:, 7:]

    d = d.sample(frac=1)
    d_train = d[:400]
    d_test = d[400:]

    d_train_att = d_train.drop(['psize'], axis=1)
    d_train_pass = d_train['psize']

    d_test_att = d_test.drop(['psize'], axis=1)
    d_test_pass = d_test['psize']

    d_att = d.drop(['psize'], axis=1)
    d_pass = d['psize']
    return d_train_att,d_train_pass,d_test_att,d_test_pass,d_att,d_pass

def training(d_train_att,d_train_pass):
    t = RandomForestClassifier(max_features=1, random_state=0, n_estimators=100)
    t = t.fit(d_train_att, d_train_pass)
    return t

def testing(t,testdataframe):
    return t.predict(testdataframe)