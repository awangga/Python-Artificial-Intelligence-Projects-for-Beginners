# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 11:01:15 2021

@author: Dinda Anik M
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preparation(datasetpath):
    d = pd.read_csv(datasetpath,
                    sep=',', header=None, error_bad_lines=False,
                    warn_bad_lines=False, usecols=[0, 1, 2, 3, 4, 5, 6],
                    names=['admit', 'gre', 'gpa', 'ses', 'Gender_Male', 'Race', 'rank' ])

    d = pd.get_dummies(d, columns=['gre', 'gpa', 'ses', 'Gender_Male', 'Race', 'rank'])

    encode = LabelEncoder()
    d['admit'] = encode.fit_transform(d['admit'])

    d = d.sample(frac=1)

    df_att = d.iloc[:, 1:117]
    df_label = d.iloc[:, 0:1]

    df_train_att = df_att[:75]
    df_train_label = df_label[:75]
    df_test_att = df_att[75:]
    df_test_label = df_label[75:]

    df_train_label = df_train_label['admit']
    df_test_label = df_test_label['admit']

    return df_train_att, df_train_label, df_test_att, df_test_label, df_att, df_label


def training(df_train_att, df_train_label):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(max_features=50, random_state=0, n_estimators=100)
    clf = clf.fit(df_train_att, df_train_label)
    return clf


def testing(clf, df_test_att):
    return clf.predict(df_test_att.head())