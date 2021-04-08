# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 06:58:29 2021

@author: Tuf Gaming
"""

import pandas
from sklearn.ensemble import RandomForestClassifier

def preparation():
    yuki = pandas.read_csv('Chapter01/dataset/heart.txt', sep='\s+', header=None, error_bad_lines=False, warn_bad_lines=False,
                    names=['age', 'sex', 'cp', 'trestbps', 
                            'chol', 'fbs', 'restecg', 'thalach', 'exang', 
                            'oldpeak', 'slope', 'ca', 'thal', 'target'])
    yuki = yuki.sample(frac=1)
    
    yuki_attribute = yuki.iloc[:, :13]
    yuki_var = yuki.iloc[:, 13:]
    
    yuki_train_attribute = yuki_attribute[:152]
    yuki_train_var = yuki_var[:152]
    yuki_test_attribute = yuki_attribute[152:]
    yuki_test_var = yuki_var[152:]

    yuki_train_var = yuki_train_var['target']
    yuki_test_var = yuki_test_var['target']
    
    data = [[yuki_train_attribute,yuki_train_var], [yuki_test_attribute, yuki_test_var]]
    return data

def training(yuki_train_attribute, yuki_train_var):
    ardi = RandomForestClassifier(max_features=2, random_state=0, n_estimators=100)
    ardi = ardi.fit(yuki_train_attribute, yuki_train_var)
    return ardi

def testing(ardi, testdataframe):
    return ardi.predict(testdataframe)