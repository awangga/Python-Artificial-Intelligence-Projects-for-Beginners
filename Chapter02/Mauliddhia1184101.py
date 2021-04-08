import pandas as pd
from sklearn import preprocessing

def preparation(datasetpath):
    f = pd.read_csv(datasetpath, sep=',', header=None, error_bad_lines=False,
                    warn_bad_lines=False, usecols=[0,1,2,3,4,5,6,7,8,9,10,
                                                   11,12,13,14,15,16,17,18,19,20,
                                                   21,22,23,24,25,26,27,28,29,30,
                                                   31,32,33,34,35,36], names=['a1','a2','a3','a4','a5','a6',
                                                        'b1','b2','b3','b4','b5','b6',
                                                        'c1','c2','c3','c4','c5','c6',
                                                        'd1','d2','d3','d4','d5','d6',
                                                        'e1','e2','e3','e4','e5','e6',
                                                        'f1','f2','f3','f4','f5','f6',
                                                        'Class'])

    f = pd.get_dummies(f, columns=['a1','a2','a3','a4','a5','a6',
                                   'b1','b2','b3','b4','b5','b6',
                                   'c1','c2','c3','c4','c5','c6',
                                   'd1','d2','d3','d4','d5','d6',
                                   'e1','e2','e3','e4','e5','e6',
                                   'f1','f2','f3','f4','f5','f6'])

    encode = preprocessing.LabelEncoder()
    f['Class'] = encode.fit_transform(f['Class'])

    f= f.sample(frac=1)

    f_att = f.iloc[:, 1:200]
    f_label = f.iloc[:, 0:10]

    f_train_att = f_att[:3000]
    f_train_label = f_label[:3000]
    f_test_att = f_att[3000:]
    f_test_label = f_label[3000:]

    f_train_label = f_train_label['Class']
    f_test_label = f_test_label['Class']

    return f_train_att, f_train_label, f_test_att, f_test_label, f_att, f_label

def training(f_train_att, f_train_label):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(max_features=10, n_jobs=-1, random_state=36, n_estimators=100)
    clf = clf.fit(f_train_att, f_train_label)
    return clf


def testing(clf, f_test_att):
    return clf.predict(f_test_att.head())