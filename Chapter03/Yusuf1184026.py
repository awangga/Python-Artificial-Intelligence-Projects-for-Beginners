# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 05:24:22 2021

@author: HP
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

def preparation():
    d = pd.read_csv('Chapter01/dataset/Kyoto_Restaurant_Info.csv')
    vc = CountVectorizer()
    d['res'] = d.apply(lambda row: 1 if row['TotalRating'] >= 3.2 else 0, axis=1)
    d = d.sample(frac=1)
    d = [d[:int(len(d)*0.75)], d[int(len(d)*0.75):]]
    data = [[vc.fit_transform(d[0]['Name']),d[0]['res']],[vc.transform(d[1]['Name']),d[1]['res']]]
    return data

def training(trainAttr, trainVar):
    t = RandomForestClassifier(n_estimators=80)
    t = t.fit(trainAttr, trainVar)
    return t

def testing(t, testAttr):
    return t.predict(testAttr)