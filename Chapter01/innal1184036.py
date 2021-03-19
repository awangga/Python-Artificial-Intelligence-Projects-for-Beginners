from sklearn import tree
from sklearn.model_selection import cross_val_score

import pandas

def preparation():
    data_frame_source = pandas.read_csv('Chapter01/dataset/phishing.csv', sep=',')
    data_frame_source = data_frame_source.sample(frac=1)
    data_frame_source_train = data_frame_source[:5526]
    data_frame_source_test = data_frame_source[5526:]
    data_frame_source_train_attribute = data_frame_source_train.drop(['class'], axis=1)
    data_frame_source_train_class = data_frame_source_train['class']
    data_frame_source_test_attribute = data_frame_source_test.drop(['class'], axis=1)
    data_frame_source_test_class = data_frame_source_test['class']
    data = [[data_frame_source_train_attribute,data_frame_source_train_class], [data_frame_source_test_attribute,data_frame_source_test_class]]
    return data

def training(data_frame_source_train_att, data_frame_source_train_class):
    t = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
    t = t.fit(data_frame_source_train_att,data_frame_source_train_class)
    return t

def testing(t, testdataframe):
    return t.predict(testdataframe)