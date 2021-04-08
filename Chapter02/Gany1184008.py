# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 14:03:48 2021

@author: GANY
"""
import pandas

def preparation(datasetpath):
    tes = pandas.read_csv(datasetpath, sep='\s+', header=None, error_bad_lines=False, warn_bad_lines=False, names=['CO(GT)','PT08.S1(CO)','NMHC(GT)','C6H6(GT)','(NMHC)','GT','PT08.S3(Nox)','NO2(GT)','PT08.S4(NO2)','PT08.S5(O3)','T','RH'])
        
    tes_train_att = tes.iloc[:, :11]
    tes_train_pass = tes.iloc[:, 11:]    
    
    tes_train = tes[:4678]
    tes_test = tes[4678:]
    #data pertama
    tes_train_att = tes_train.drop(['RH'], axis=1)
    tes_train_pass = tes_train['RH'] 
    #kemudian dikurang
    tes_test_att = tes_test.drop(['RH'], axis=1) 
    tes_test_pass = tes_test['RH']
    #total data
    tes_att = tes.drop(['RH'], axis=1)
    tes_pass = tes['RH']
    
    return tes_train_att,tes_train_pass,tes_test_att,tes_test_pass,tes_att,tes_pass
    

def training(tes_train_attribute, tes_train_var):
    from sklearn.ensemble import RandomForestClassifier
    x = RandomForestClassifier(max_features=2, random_state=0, n_estimators=100)
    x = x.fit(tes_train_attribute,tes_train_var)
    return x

def testing(x,tes_test_attribute):
    return x.predict(tes_test_attribute.head())
