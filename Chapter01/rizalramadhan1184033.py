from sklearn import tree 
import pandas as pd 

def prepoc(datapath):
    nanas = pd.read_csv(datapath, sep=',') 
    len(nanas) 

    nanas = nanas.sample(frac=1) 
    nanas_train = nanas[:201] 
    nanas_test = nanas[200:] 

    nanas_train_att = nanas_train.drop(['blue'], axis=1) 
    nanas_train_pass = nanas_train['blue'] 
    
    nanas_test_att = nanas_test.drop(['blue'], axis=1) 
    nanas_test_pass = nanas_test['blue'] 
    
    nanas_att = nanas.drop(['blue'], axis=1) 
    nanas_pass = nanas['blue'] 
    return nanas_train_att,nanas_train_pass,nanas_test_att,nanas_test_pass,nanas_att,nanas_pass 

def training(nanas_train_att, nanas_train_pass): 
    apel = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5) 
    apel = apel.fit(nanas_train_att, nanas_train_pass)
    return apel

def testing(apel,testdataframe): 
    return apel.predict(testdataframe)