# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 11:48:38 2021

@author: ASUS
"""

import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer


def preparation():
    dfs = pandas.read_csv('Chapter01/dataset/flipkart.csv', sep= ',')

    
    cv = CountVectorizer()
    dfs = dfs.sample(frac=1)
    dfs_train = dfs[:4988]
    dfs_test = dfs[4988:]
    dfs_train_attribute = cv.fit_transform(dfs_train['review'])
    dfs_train_win = dfs_train['rating']
    dfs_test_attribute = cv.transform(dfs_test['review'])
    dfs_test_win = dfs_test['rating']
    data = [[dfs_train_attribute,dfs_train_win], [dfs_test_attribute, dfs_test_win]]
    return data

def training(dfs_train_att, dfs_train_win):
    t = RandomForestClassifier(n_estimators=100)
    t = t.fit(dfs_train_att,dfs_train_win)
    return t

def testing(t, testdataframe):
    return t.predict(testdataframe)