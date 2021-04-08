# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 09:48:10 2021

@author: Nurul Kamila (1184038)
"""

import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer


def preparation():
    dfs = pandas.read_csv('Chapter01/dataset/dataset-of-00s.csv', sep= ',')


    cv = CountVectorizer()
    dfs = dfs.sample(frac=1)
    dfs_train = dfs[:2937]
    dfs_test = dfs[2937:]
    dfs_train_attribute = cv.fit_transform(dfs_train['track'])
    dfs_train_win = dfs_train['target']
    dfs_test_attribute = cv.transform(dfs_test['track'])
    dfs_test_win = dfs_test['target']
    data = [[dfs_train_attribute,dfs_train_win], [dfs_test_attribute, dfs_test_win]]
    return data

def training(dfs_train_att, dfs_train_win):
    t = RandomForestClassifier(n_estimators=100)
    t = t.fit(dfs_train_att,dfs_train_win)
    return t

def testing(t, testdataframe):
    return t.predict(testdataframe) 