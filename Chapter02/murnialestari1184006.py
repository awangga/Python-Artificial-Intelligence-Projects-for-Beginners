# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 11:50:41 2021

@author: Murnia
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preparation(datasetpath):
    d = pd.read_csv(datasetpath,
                    sep=',', header=None, error_bad_lines=False,
                    warn_bad_lines=False, usecols=[0, 1, 2, 3, 4],
                    names=['Variance', 'Skewness', 'Curtosis', 'Entropy', 'Class' ])

    d = pd.get_dummies(d, columns=['Variance', 'Skewness', 'Curtosis', 'Entropy', ])

    encode = LabelEncoder()
    d['Class'] = encode.fit_transform(d['Class'])

    d = d.sample(frac=1)

    df_att = d.iloc[:, 1:132]
    df_label = d.iloc[:, 0:1]

    df_train_att = df_att[:50]
    df_train_label = df_label[:50]
    df_test_att = df_att[50:]
    df_test_label = df_label[50:]

    df_train_label = df_train_label['Class']
    df_test_label = df_test_label['Class']

    return df_train_att, df_train_label, df_test_att, df_test_label, df_att, df_label


def training(df_train_att, df_train_label):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(max_features=1, random_state=0, n_estimators=100)
    clf = clf.fit(df_train_att, df_train_label)
    return clf


def testing(clf, df_test_att):
    return clf.predict(df_test_att.head())