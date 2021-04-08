import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

def preparation():
    d = pd.read_csv('Chapter01/dataset/Womens Clothing E-Commerce Reviews.csv')
    vc = CountVectorizer()
    arr = []
    d['lov'] = d.apply(lambda row: 1 if row['Rating'] >= 2 else 0, axis=1)
    d = d.sample(frac=1)
    d = [d[:int(len(d)*0.75)], d[int(len(d)*0.75):]]
    data = [[vc.fit_transform(d[0]['Review Text'].values.astype('U')),d[0]['lov']],[vc.transform(d[1]['Review Text'].values.astype('U')),d[1]['lov']]]
    return data

def training(trainAttr, trainVar):
    t = RandomForestClassifier(n_estimators=80)
    t = t.fit(trainAttr, trainVar)
    return t

def testing(t, testAttr):
    return t.predict(testAttr)