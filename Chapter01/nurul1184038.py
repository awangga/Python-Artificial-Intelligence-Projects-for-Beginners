from sklearn import tree
import pandas as pd


def prepoc(datapath):
    d = pd.read_csv(datapath, sep=',')
    len(d)

    # shuffle data
    d = d.sample(frac=1)
    d_train = d[:350]
    d_test = d[350:]

    d_train_att = d_train.drop(['pass'], axis=1)
    d_train_pass = d_train['pass']

    d_test_att = d_test.drop(['pass'], axis=1)
    d_test_pass = d_test['pass']

    d_att = d.drop(['pass'], axis=1)
    d_pass = d['pass']
    return d_train_att, d_train_pass, d_test_att, d_test_pass, d_att, d_pass


def training(d_train_att, d_train_pass):
    t = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
    t = t.fit(d_train_att, d_train_pass)
    return t

def testing(t, testdataframe):
    return t.predict(testdataframe)