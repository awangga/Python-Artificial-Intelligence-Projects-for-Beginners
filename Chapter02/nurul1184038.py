# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 11:33:20 2021

@author: Nurul Kamila (1184038)
"""
import pandas
from sklearn.ensemble import RandomForestClassifier

def preparation():
    d = pandas.read_csv('Chapter01/dataset/prediksi-siswa-lulus-atau-gagal.txt', sep='\s+', header=None, error_bad_lines=False, warn_bad_lines=False,
                     names=['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3', 'pass'])
    d = d.sample(frac=1)

    d_attribute = d.iloc[:, :29]
    d_var = d.iloc[:, 29:]

    d_train_attribute = d_attribute[:197]
    d_train_var = d_var[:197]
    d_test_attribute = d_attribute[197:]
    d_test_var = d_var[197:]

    d_train_var = d_train_var['pass']
    d_test_var = d_test_var['pass']

    data = [[d_train_attribute,d_train_var], [d_test_attribute, d_test_var]]
    return data

def training(d_train_attribute, d_train_var):
    t = RandomForestClassifier(max_features=2, random_state=0, n_estimators=100)
    t = t.fit(d_train_attribute, d_train_var)
    return t

def testing(t, testdataframe):
    return t.predict(testdataframe)

