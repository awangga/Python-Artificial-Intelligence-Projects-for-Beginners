import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

def prepoc(datapath):
    nanas = pd.read_csv(datapath, sep=',')
    len(nanas)
    
    cv = CountVectorizer()
    # shuffle data
    
    nanas = nanas.sample(frac=1)
    nanas_train = nanas[:1000]
    nanas_test = nanas[1000:]

    nanas_train_att = cv.fit_transform(nanas_train['text'])
    nanas_train_pass = nanas_train['spam']
    
    nanas_test_att = cv.transform(nanas_test['text'])
    nanas_test_pass = nanas_test['spam']
    

    return nanas_train_att,nanas_train_pass,nanas_test_att,nanas_test_pass

def training(nanas_train_att,nanas_train_pass):
    apel = RandomForestClassifier(n_estimators=100)
    apel = apel.fit(nanas_train_att, nanas_train_pass)
    return apel

def testing(apel,testdataframe):
    return apel.predict(testdataframe)

