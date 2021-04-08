# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 22:58:25 2021

@author: Muh Amri
"""

import pandas 
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

def preparation():
    buy = pandas.read_csv('Chapter01/dataset/final_data.csv', sep=',')
    
    cv = CountVectorizer()
    buy = buy.sample(frac=1)
    buy_train = buy[:1000]
    buy_test = buy[1000:]
    
    buy_train_attribute = cv.fit_transform(buy_train['text'])
    buy_train_win = buy_train['status']
    
    buy_test_attribute = cv.transform(buy_test['text'])
    buy_test_win = buy_test['status']
    
    data = [[buy_train_attribute,buy_train_win], [buy_test_attribute,buy_test_win]]
    return data

def training(buy_train_attribute,buy_train_win):
    w = RandomForestClassifier(n_estimators=100)
    w = w.fit(buy_train_attribute,buy_train_win)
    return w

def testing (w, testdataframe):
    return w.predict(testdataframe)
    