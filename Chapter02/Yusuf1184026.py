# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 09:07:35 2021

@author: HP
"""

import pandas as pd

def preparation(dataset):
    d = pd.read_csv(dataset,
                    sep=',', header=None, error_bad_lines=False,
                    warn_bad_lines=False, usecols=[0, 1, 2, 3, 4,5],
                    names=['buying' ,'maint', 'doors','persons','lug_boot','safety'])
    
    # one-hot encoding pada semua kolom
    d = pd.get_dummies(d, columns=['buying' ,'maint', 'doors','persons','lug_boot','safety'])
    
    # menambahkan kolom class dan generate label, nilai 0(unacc), 1(acc), 2(good) dan 3(vgood) 
    d['class'] = d.apply(lambda row: 0 if (row['buying_low']+row['maint_low']+
                                            row['doors_2']+row['persons_2']+row['lug_boot_big']+
                                            row['safety_high']) <= 1 else
                          (1 if(row['buying_low']+row['maint_low']+
                                            row['doors_2']+row['persons_2']+row['lug_boot_big']+
                                            row['safety_high']) <= 3 else 
                           (2 if(row['buying_low']+row['maint_low']+
                                            row['doors_2']+row['persons_2']+row['lug_boot_big']+
                                            row['safety_high']) <= 5 else 3 )), axis=1)
    # shuffle data
    d_shuffle = d.sample(frac=1)
    # memetakan atribut dan label
    df_att = d_shuffle.iloc[:, :21]
    df_label = d_shuffle.iloc[:, 21:]
    # banyak data yang akan ditraining
    percent_training = int(len(d)*0.75)
    # data train
    df_train_att = df_att[:percent_training]
    df_train_label = df_label[:percent_training]
    # data test
    df_test_att = df_att[percent_training:]
    df_test_label = df_label[percent_training:]

    df_train_label = df_train_label['class']
    df_test_label = df_test_label['class']

    return df_train_att, df_train_label, df_test_att, df_test_label, df_att, df_label


def training(df_train_att, df_train_label):
    from sklearn.ensemble import RandomForestClassifier
    # instansiasi variabel klasifikasi dengan metode random forest classifier, max atribut yang digunakan ialah 4 kolom di setiap independent treenya
    clf = RandomForestClassifier(max_features=4, random_state=0, n_estimators=100)
    # klasifikasi data training df_train_att dan df_train_label
    clf = clf.fit(df_train_att, df_train_label)
    return clf

def testing(clf, df_test_att):
    return clf.predict(df_test_att.head())