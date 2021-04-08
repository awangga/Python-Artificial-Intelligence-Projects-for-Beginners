# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 21:55:05 2021

@author: Asus
"""

import pandas
from sklearn.ensemble import RandomForestClassifier

def preparation():
    d = pandas.read_csv('Chapter01/dataset/kanker_payudara.txt', sep='\s+', header=None, error_bad_lines=False, warn_bad_lines=False,
                     names=['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area ', 'mean_smoothness', 'diagnosis'])
    d = d.sample(frac=1)
    
    d_attribute = d.iloc[:, :5]
    d_var = d.iloc[:, 5:]

    d_train_attribute = d_attribute[:284]
    d_train_var = d_var[:284]
    d_test_attribute = d_attribute[284:]
    d_test_var = d_var[284:]

    d_train_var = d_train_var['diagnosis']
    d_test_var = d_test_var['diagnosis']

    data = [[d_train_attribute,d_train_var], [d_test_attribute, d_test_var]]
    return data

def training(d_train_attribute, d_train_var):
    t = RandomForestClassifier(max_features=2, random_state=0, n_estimators=100)
    t = t.fit(d_train_attribute, d_train_var)
    return t

def testing(t, testdataframe):
    return t.predict(testdataframe)