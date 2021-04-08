# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 20:32:35 2021

@author: USER
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

def preparation(datasetpath):
    d = pd.read_csv(datasetpath, sep=";")
    len(d)
    
    cv = CountVectorizer()
    # shuffle data
    
    d = d.sample(frac=1)
    d_train = d[:5403]
    d_test = d[5403:]

    d_train_att = cv.fit_transform(d_train['Tweet'])
    d_train_pass = d_train['sentimen']
    
    d_test_att = cv.transform(d_test['Tweet'])
    d_test_pass = d_test['sentimen']
    

    return d_train_att,d_train_pass,d_test_att,d_test_pass

def training(d_train_att,d_train_pass):
    t = RandomForestClassifier(n_estimators=100)
    t = t.fit(d_train_att, d_train_pass)
    return t

def testing(t,testdataframe):
    return t.predict(testdataframe)