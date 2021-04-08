# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 11:21:10 2021

@author: USER
"""

import pandas as pd

def preparation(datasetpath):
    d = pd.read_csv(datasetpath, sep='\s+', header=None, error_bad_lines=False, warn_bad_lines=False, names=['year','month','day','order','country','session ID','page 1 (main category)','colour','location','model photography','price','price 2','page'])

     # shuffle data
    d_shuffle = d.sample(frac=1)
    # memetakan atribut dan label
    df_att = d_shuffle.iloc[:, :12]
    df_label = d_shuffle.iloc[:, 12:]
    # banyak data yang akan ditraining
    percent_training = int(len(d)*0.75)
    # data train
    df_train_att = df_att[:percent_training]
    df_train_label = df_label[:percent_training]
    # data test
    df_test_att = df_att[percent_training:]
    df_test_label = df_label[percent_training:]

    df_train_label = df_train_label['page']
    df_test_label = df_test_label['page']

    return df_train_att, df_train_label, df_test_att, df_test_label, df_att, df_label


def training(df_train_att, df_train_label):
    from sklearn.ensemble import RandomForestClassifier
    # instansiasi variabel klasifikasi dengan metode random forest classifier, max atribut yang digunakan ialah 3 kolom di setiap independent treenya
    clf = RandomForestClassifier(max_features=3, random_state=0, n_estimators=100)
    # klasifikasi data training df_train_att dan df_train_label
    clf = clf.fit(df_train_att, df_train_label)
    return clf

def testing(clf, df_test_att):
    return clf.predict(df_test_att.head())