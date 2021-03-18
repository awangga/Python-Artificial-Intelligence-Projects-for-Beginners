from sklearn import tree
from sklearn.model_selection import cross_val_score
import pandas


def prepoc(datapath):
    x = pandas.read_csv(datapath, sep=',')
    len(x)
    
    # shuffle data
    dd = x.sample(frac=1)  # frac=1 (means 100% data shuffle)
    data_train = dd[:5000] # menampung 300 data pertama yang ditampung di variabel dd_train
    data_test = dd[500:] 

    data_train_new = data_train.drop(['Total Deaths'], axis=1)
    data_train_label = data_train['Total Deaths']
    
    data_test_new = data_test.drop(['Total Deaths'], axis=1)
    data_test_label = data_test['Total Deaths']
    
    d_new = dd.drop(['Total Deaths'], axis=1)
    d_label = dd['Total Deaths'
               ]
    return data_train_new,data_train_label,data_test_new,data_test_label,d_new,d_label

#Training
def training(data_train_new, data_train_label):
    #decision Tree
    dt = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
    dt = dt.fit(data_train_new, data_train_label)
    scores = cross_val_score(dt, data_train_new, data_train_label, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return dt

#Predict
def predict(dt, data_testing):
    return dt.predict(data_testing)