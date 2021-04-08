import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def preparation(datapath):
    d = pd.read_csv('Chapter01/dataset/phishing.txt', sep='\s+', header=None, error_bad_lines=False, warn_bad_lines=False, names=['Index', 'UsingIP', 'LongURL', 'ShortURL', 'Symbol@', 'Redirecting//', 'PrefixSuffix-', 'SubDomains', 'HTTPS', 'DomainRegLen', 'Favicon', 'NonStdPort', 'HTTPSDomainURL', 'RequestURL', 'AnchorURL', 'LinksInScriptTags', 'ServerFormHandler', 'InfoEmail', 'AbnormalURL', 'WebsiteForwarding', 'StatusBarCust', 'DisableRightClick', 'UsingPopupWindow', 'IframeRedirection', 'AgeofDomain', 'DNSRecording', 'WebsiteTraffic', 'PageRank', 'GoogleIndex', 'LinksPointingToPage', 'StatsReport', 'class'])
    len(d)

    # shuffle data
    d = d.sample(frac=1)
    d_train = d[:5250]
    d_test = d[5250:]

    d_train_att = d_train.drop(['class'], axis=1)
    d_train_pass = d_train['class']

    d_test_att = d_test.drop(['class'], axis=1)
    d_test_pass = d_test['class']

    d_att = d.drop(['class'], axis=1)
    d_pass = d['class']
    return d_train_att,d_train_pass,d_test_att,d_test_pass,d_att,d_pass

def training(d_train_att,d_train_pass):
    t = RandomForestClassifier(max_features=1, random_state=0, n_estimators=100)
    t = t.fit(d_train_att, d_train_pass)
    return t

def testing(t,testdataframe):
    return t.predict(testdataframe)