# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 12:46:48 2021

@author: lovo
"""
import pandas
from sklearn.ensemble import RandomForestClassifier

def preparation():
    d = pandas.read_csv('Chapter01/dataset/zoo.txt', sep='\s+', header=None, error_bad_lines=False, warn_bad_lines=False,
                     names=['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic','catsize','class_type'
    ])
    d = d.sample(frac=1)

    d_attribute = d.iloc[:, :16]
    d_var = d.iloc[:, 16:]

    d_train_attribute = d_attribute[:50]
    d_train_var = d_var[:50]
    d_test_attribute = d_attribute[50:]
    d_test_var = d_var[50:]

    d_train_var = d_train_var['class_type']
    d_test_var = d_test_var['class_type']

    data = [[d_train_attribute,d_train_var], [d_test_attribute, d_test_var]]
    return data

def training(d_train_attribute, d_train_var):
    t = RandomForestClassifier(max_features=2, random_state=0, n_estimators=100)
    t = t.fit(d_train_attribute, d_train_var)
    return t

def testing(t, testdataframe):
    return t.predict(testdataframe)

