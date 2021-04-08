# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 20:07:42 2021

@author: Dian
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def prepoc(datapath):
    d = pd.read_csv(datapath, sep='\s+', header=None, error_bad_lines=False, warn_bad_lines=False, names=['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'duration', 'campaign', 'poutcome'])

    d_train_att = d.iloc[:, :12]
    d_train_pass = d.iloc[:, 12:]

    d = d.sample(frac=1)
    d_train = d[:4119]
    d_test = d[4119:]

    d_train_att = d_train.drop(['poutcome'], axis=1)
    d_train_pass = d_train['poutcome']

    d_test_att = d_test.drop(['poutcome'], axis=1)
    d_test_pass = d_test['poutcome']

    d_att = d.drop(['poutcome'], axis=1)
    d_pass = d['poutcome']
    return d_train_att,d_train_pass,d_test_att,d_test_pass,d_att,d_pass

def training(d_train_att,d_train_pass):
    t = RandomForestClassifier(max_features=1, random_state=0, n_estimators=100)
    t = t.fit(d_train_att, d_train_pass)
    return t

def testing(t,testdataframe):
    return t.predict(testdataframe)