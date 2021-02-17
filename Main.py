import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def Open_DataSet():
    df = pd.read_csv("emails.csv")
    X = df.iloc[:,1:3001].values
    Y = df.iloc[:,-1].values
    return X,Y

def NaiveBayes_Simple(X,Y,valueTest_size):
    train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size = valueTest_size)
    mnb = MultinomialNB(alpha=1)
    mnb.fit(train_x,train_y)
    y_pred1 = mnb.predict(test_x)
    print("Accuracy Score for Naive Bayes Simple: ", accuracy_score(y_pred1,test_y))

def NaiveBayes_K_folds(X,Y):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=10)
    kf.get_n_splits(X)
    cv_scores = list()
    indexFolds = 1
    for train_index, test_index in kf.split(X,Y):
        print('-----------------------------------------------------------------')
        print('Fold ' + str(indexFolds))
        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = Y[train_index], Y[test_index]
        mnb = MultinomialNB(alpha=1)
        mnb.fit(X_train,y_train)
        y_pred1 = mnb.predict(X_val)
        print("Score: ", accuracy_score(y_pred1,y_val))
        cv_scores.append(accuracy_score(y_pred1,y_val))
        indexFolds = indexFolds + 1
    print('Estimated Accuracy %.3f (%.3f)' % (np.mean(cv_scores), np.std(cv_scores)))

def SVM(X,Y,valueTest_size):
    train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size = valueTest_size)
    svc = SVC(C=1.0,kernel='rbf',gamma='auto')
    svc.fit(train_x,train_y)
    y_pred2 = svc.predict(test_x)
    print("Accuracy Score for SVC : ", accuracy_score(y_pred2,test_y))

def SVM_K_folds(X,Y):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=10)
    kf.get_n_splits(X)
    cv_scores = list()
    indexFolds = 1
    for train_index, test_index in kf.split(X,Y):
        print('-----------------------------------------------------------------')
        print('Fold ' + str(indexFolds))
        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = Y[train_index], Y[test_index]
        svc = SVC(C=1.0,kernel='rbf',gamma='auto')
        svc.fit(X_train,y_train)
        y_pred2 = svc.predict(X_val)
        print("Score: ", accuracy_score(y_pred2,y_val))
        cv_scores.append(accuracy_score(y_pred2,y_val))
        indexFolds = indexFolds + 1
    print('Estimated Accuracy %.3f (%.3f)' % (np.mean(cv_scores), np.std(cv_scores)))

def RandomForest(X,Y,valueTest_size):
    train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size = valueTest_size)
    rfc = RandomForestClassifier(n_estimators=10000,criterion='gini')
    rfc.fit(train_x,train_y)
    y_pred3 = rfc.predict(test_x)
    print("Accuracy Score of Random Forest Classifier : ", accuracy_score(y_pred3,test_y))

def RandomForest_K_folds(X,Y):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=10)
    kf.get_n_splits(X)
    cv_scores = list()
    indexFolds = 1
    for train_index, test_index in kf.split(X,Y):
        print('-----------------------------------------------------------------')
        print('Fold ' + str(indexFolds))
        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = Y[train_index], Y[test_index]
        rfc = RandomForestClassifier(n_estimators=100,criterion='gini')
        rfc.fit(X_train,y_train)
        y_pred3 = rfc.predict(X_val)
        print("Score: ", accuracy_score(y_pred3,y_val))
        cv_scores.append(accuracy_score(y_pred3,y_val))
        indexFolds = indexFolds + 1
    print('Estimated Accuracy %.3f (%.3f)' % (np.mean(cv_scores), np.std(cv_scores)))

X,Y = Open_DataSet()
# RandomForest(X,Y,0.25)
# SVM(X,Y,0.25)
# NaiveBayes_Simple(X,Y,0.25)
RandomForest_K_folds(X,Y)
# SVM_K_folds(X,Y)
# NaiveBayes_K_folds(X,Y)

