import pandas as pd
from sklearn import preprocessing

def preparation(datasetpath):
    f = pd.read_csv(datasetpath, sep=',', header=None, error_bad_lines=False,
                    warn_bad_lines=False, usecols=[0,1,2,3,4,5,6,7,8,9,10,
                                                   11,12,13,14,15,16], names=['Class','handicapped-infants','water-project-cost-sharing','adoption-of-the-budget-resolution','physician-fee-freeze','el-salvador-aid',
                                                        'religious-groups-in-schools','anti-satellite-test-ban','aid-to-nicaraguan-contras','mx-missile','immigration','synfuels-corporation-cutback',
                                                        'education-spending','superfund-right-to-sue','crime','duty-free-exports','export-administration-act-south-africa'])

    f = pd.get_dummies(f, columns=['handicapped-infants','water-project-cost-sharing','adoption-of-the-budget-resolution','physician-fee-freeze','el-salvador-aid',
                                                        'religious-groups-in-schools','anti-satellite-test-ban','aid-to-nicaraguan-contras','mx-missile','immigration','synfuels-corporation-cutback',
                                                        'education-spending','superfund-right-to-sue','crime','duty-free-exports','export-administration-act-south-africa'])


    encode = preprocessing.LabelEncoder()
    f['Class'] = encode.fit_transform(f['Class'])

    f= f.sample(frac=1)

    f_att = f.iloc[:, 1:200]
    f_label = f.iloc[:, 0:10]

    f_train_att = f_att[:400]
    f_train_label = f_label[:400]
    f_test_att = f_att[400:]
    f_test_label = f_label[400:]

    f_train_label = f_train_label['Class']
    f_test_label = f_test_label['Class']

    return f_train_att, f_train_label, f_test_att, f_test_label, f_att, f_label

def training(f_train_att, f_train_label):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(max_features=10, n_jobs=-1, random_state=17, n_estimators=100)
    clf = clf.fit(f_train_att, f_train_label)
    return clf


def testing(clf, f_test_att):
    return clf.predict(f_test_att.head())