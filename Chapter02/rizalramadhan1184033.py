import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def prepoc(datapath):
    nanas = pd.read_csv(datapath, sep=',')
    len(nanas)

    # shuffle data
    nanas_train_att = nanas.iloc[:, :20]
    nanas_train_pass = nanas.iloc[:, 20:]

    nanas = nanas.sample(frac=1)
    nanas_train = nanas[:1000]
    nanas_test = nanas[1000:]

    nanas_train_att = nanas_train.drop(['price_range'], axis=1)
    nanas_train_pass = nanas_train['price_range']

    nanas_test_att = nanas_test.drop(['price_range'], axis=1)
    nanas_test_pass = nanas_test['price_range']

    nanas_att = nanas.drop(['price_range'], axis=1)
    nanas_pass = nanas['price_range']
    return nanas_train_att,nanas_train_pass,nanas_test_att,nanas_test_pass,nanas_att,nanas_pass

def training(nanas_train_att,nanas_train_pass):
    apel = RandomForestClassifier(max_features=1, random_state=0, n_estimators=100)
    apel = apel.fit(nanas_train_att, nanas_train_pass)
    return apel

def testing(apel,testdataframe):
    return apel.predict(testdataframe)

