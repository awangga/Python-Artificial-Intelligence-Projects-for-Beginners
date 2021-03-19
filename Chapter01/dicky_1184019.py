# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 21:11:32 2021

@author: ASUS iD
"""

from sklearn import tree
import pandas as pd

def prepoc(datapath):
    d = pd.read_csv(datapath, sep=',')
    len(d)
    
    # shuffle data
    d = d.sample(frac=1)
    d_train = d[:300]
    d_test = d[300:]
    #300 data 
    
    d_train_att = d_train.drop(['Fatalities'], axis=1)
    d_train_pass = d_train['Fatalities']
    #data setelah diproses
    d_test_att = d_test.drop(['Fatalities'], axis=1)
    d_test_pass = d_test['Fatalities']
    
    d_att = d.drop(['Fatalities'], axis=1)
    d_pass = d['Fatalities']
    return d_train_att,d_train_pass,d_test_att,d_test_pass,d_att,d_pass
    
def training(d_train_att,d_train_pass):
    t = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
    t = t.fit(d_train_att, d_train_pass)
    return t

def testing(t,testdataframe):
    return t.predict(testdataframe)