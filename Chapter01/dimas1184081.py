# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 23:02:50 2021

@author: ASUS
"""

from sklearn import tree
import pandas as pd

def prepoc(datapath):
    anak = pd.read_csv(datapath, sep=',')
    len(anak)

    # shuffle data
    anak = anak.sample(frac=1)
    anak_train = anak[:1045]
    anak_test = anak[522:]

    anak_train_att = anak_train.drop(['generation'], axis=1)
    anak_train_pass = anak_train['generation']

    anak_test_att = anak_test.drop(['generation'], axis=1)
    anak_test_pass = anak_test['generation']

    anak_att = anak.drop(['generation'], axis=1)
    anak_pass = anak['generation']
    return anak_train_att,anak_train_pass,anak_test_att,anak_test_pass,anak_att,anak_pass

def training(anak_train_att,anak_train_pass):
    it = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
    it = it.fit(anak_train_att, anak_train_pass)
    return it

def testing(it,testdataframe):
    return it.predict(testdataframe)