import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def preparation():
    data = pd.read_csv('Chapter01/dataset/coviddd.txt', sep=',', usecols=[0,1,2,3,4,5,6], header=None, names=['umur', 'jenisKelamin', 'batuk', 'pilek', 'sesakNapas', 'inderaPenciuman', 'covid'])
   
    data = data.sample(frac=1)
    data = [data.iloc[:,:4], data.iloc[:, 4:]]


    dataAttr = data.pop(0)
    dataVar = data.pop(0)
    

    length = int(len(dataVar)*0.75)

    trainVar = dataVar[:length]
    trainAttr = dataAttr[:length]

    testVar = dataVar[length:]
    testAttr = dataAttr[length:]

    return [[trainAttr, trainVar], [testAttr, testVar]]

def training(trainAttr, trainVar):
    t = RandomForestClassifier(max_features=4, random_state=0, n_estimators=100)
    t = t.fit(trainAttr, trainVar)
    return t

def testing(t, testAttr):
    return t.predict(testAttr)