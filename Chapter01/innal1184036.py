from sklearn import tree
from sklearn.model_selection import cross_val_score

import pandas

def preparation():
    dfs = pandas.read_csv('Chapter01/dataset/phishing.csv', sep=',')
    dfs = dfs.sample(frac=1)
    print(len(dfs))
    dfs_train = dfs[:5526]
    dfs_test = dfs[5526:]
    dfs_train_attribute = dfs_train.drop(['class'], axis=1)
    dfs_train_class = dfs_train['class']
    dfs_test_attribute = dfs_test.drop(['class'], axis=1)
    dfs_test_class = dfs_test['class']
    data = [[dfs_train_attribute,dfs_train_class], [dfs_test_attribute,dfs_test_class]]
    return data

def training(dfs_train_att, dfs_train_class):
    t = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
    t = t.fit(dfs_train_att,dfs_train_class)
    return t

def testing(t, testdataframe):
    return t.predict(testdataframe)