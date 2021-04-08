# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 20:54:03 2021

@author: Tuf Gaming
"""

from sklearn import tree
import pandas as pd

def prepoc(datapath):
    pencak = pd.read_csv(datapath, sep=',')
    len(pencak)
    
    # shuffle data
    pencak = pencak.sample(frac=1)
    pencak_train = pencak[:1000]
    pencak_test = pencak[500:]

    pencak_train_att = pencak_train.drop(['Bankrupt'], axis=1)
    pencak_train_pass = pencak_train['Bankrupt']
    
    pencak_test_att = pencak_test.drop(['Bankrupt'], axis=1)
    pencak_test_pass = pencak_test['Bankrupt']
    
    pencak_att = pencak.drop(['Bankrupt'], axis=1)
    pencak_pass = pencak['Bankrupt']
    return pencak_train_att,pencak_train_pass,pencak_test_att,pencak_test_pass,pencak_att,pencak_pass

def training(pencak_train_att,pencak_train_pass):
    silat = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
    silat = silat.fit(pencak_train_att, pencak_train_pass)
    return silat

def testing(silat,testdataframe):
    return silat.predict(testdataframe)