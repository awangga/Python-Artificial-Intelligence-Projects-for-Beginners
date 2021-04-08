# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 23:51:57 2021

@author: ASUS
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
    d_train = d[:2500]
    d_test = d[2500:]

    d_train_att = cv.fit_transform(d_train['text'])
    d_train_pass = d_train['label_num']
    
    d_test_att = cv.transform(d_test['text'])
    d_test_pass = d_test['label_num']
    

    return d_train_att,d_train_pass,d_test_att,d_test_pass

def training(d_train_att,d_train_pass):
    t = RandomForestClassifier(n_estimators=100)
    t = t.fit(d_train_att, d_train_pass)
    return t

def testing(t,testdataframe):
    return t.predict(testdataframe)