# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 01:44:55 2021

@author: Aditya Luthfi
"""

import pandas
from sklearn.ensemble import RandomForestClassifier

def preparation():
    d = pandas.read_csv('Chapter01/dataset/iris_dataset.txt', sep='\s+', header=None, error_bad_lines=False, warn_bad_lines=False,
                     names=['sepal.length', 'sepal.width', 'petal.length', 'petal.width', 'variety'])
    d = d.sample(frac=1)
    
    d_attribute = d.iloc[:, :4]
    d_var = d.iloc[:, 4:]
    
    d_train_attribute = d_attribute[:75]
    d_train_var = d_var[:75]
    d_test_attribute = d_attribute[75:]
    d_test_var = d_var[75:]

    d_train_var = d_train_var['variety']
    d_test_var = d_test_var['variety']
    
    data = [[d_train_attribute,d_train_var], [d_test_attribute, d_test_var]]
    return data

def training(d_train_attribute, d_train_var):
    t = RandomForestClassifier(max_features=2, random_state=0, n_estimators=100)
    t = t.fit(d_train_attribute, d_train_var)
    return t

def testing(t, testdataframe):
    return t.predict(testdataframe)