# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 11:31:45 2021

@author: Dinda Anik M
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

def prepoc(datapath):
    d = pd.read_csv(datapath, sep=',')
    len(d)

    cv = CountVectorizer()
    # shuffle data

    d = d.sample(frac=1)
    d_train = d[:10000]
    d_test = d[10000:]

    d_train_att = cv.fit_transform(d_train['Review'])
    d_train_pass = d_train['Rating']

    d_test_att = cv.transform(d_test['Review'])
    d_test_pass = d_test['Rating']


    return d_train_att,d_train_pass,d_test_att,d_test_pass

def training(d_train_att,d_train_pass):
    t = RandomForestClassifier(n_estimators=100)
    t = t.fit(d_train_att, d_train_pass)
    return t

def testing(t,testdataframe):
    return t.predict(testdataframe)