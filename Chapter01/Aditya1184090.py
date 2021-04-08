# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 20:57:28 2021

@author: Aditya Luthfi
"""

import pandas
from sklearn import tree
from sklearn.model_selection import cross_val_score

def preparation():
    dfs = pandas.read_csv('Chapter01/dataset/iris_numeric_dataset.csv', sep=',')
    dfs = dfs.sample(frac=1)
    dfs_train = dfs[:75]
    dfs_test = dfs[75:]
    dfs_train_attribute = dfs_train.drop(['variety'], axis=1)
    dfs_train_win = dfs_train['variety']
    dfs_test_attribute = dfs_test.drop(['variety'], axis=1)
    dfs_test_win = dfs_test['variety']
    data = [[dfs_train_attribute,dfs_train_win], [dfs_test_attribute, dfs_test_win]]
    return data

def training(dfs_train_att, dfs_train_win):
    t = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
    t = t.fit(dfs_train_att,dfs_train_win)
    return t

def testing(t, testdataframe):
    return t.predict(testdataframe)