import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint

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
    # svc = SVC(C=1.0,kernel='rbf',gamma='auto')
    svc = SVC(C=100,kernel='rbf',gamma=0.0001)
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

def RandomizedSearch(X,Y):

    rfc = RandomForestClassifier(random_state=4)

    params = {'n_estimators': sp_randint(50,400),
              'max_features': sp_randint(2,16),
              'max_depth': sp_randint(2,10),
              'min_samples_split': sp_randint(2,25),
              'min_samples_leaf': sp_randint(1,25),
              'criterion':['gini','entropy']}

    rsearch = RandomizedSearchCV(rfc,
                                param_distributions=params,
                                n_iter=50,
                                cv=3,
                                return_train_score = True,
                                scoring='roc_auc',
                                n_jobs=None,
                                random_state=5)

    rsearch.fit(X,Y)

    print(rsearch.best_params_)

def GridSearch(X,Y):
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    clf = GridSearchCV(SVC(), tuned_parameters,
                       scoring='roc_auc')

    clf.fit(X, Y)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()

def RandomForest(X,Y,valueTest_size):
    train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size = valueTest_size)
    rfc = RandomForestClassifier(criterion='entropy', max_depth=8,max_features=13,min_samples_leaf=6,min_samples_split=11,n_estimators=230,random_state=4)
    rfc.fit(train_x,train_y)

    print("Train Results \n")
    y_train_pred  = rfc.predict(train_x)
    y_train_prob = rfc.predict_proba(train_x)[:,1]

    print("Confusion Matrix for Train : \n", confusion_matrix(train_y, y_train_pred))
    print("Accuracy Score for Train : ", accuracy_score(train_y, y_train_pred))

    print("+"*50)
    print("Test Results \n")
    y_test_pred  = rfc.predict(test_x)
    y_test_prob = rfc.predict_proba(test_x)[:,1]

    print("Confusion Matrix for Test : \n", confusion_matrix(test_y, y_test_pred))
    print("Accuracy Score for Test : ", accuracy_score(test_y, y_test_pred))
    print("ROC AUC for Test : ", roc_auc_score(test_y, y_test_prob))

    print("ROC AUC for Train : ", roc_auc_score(train_y, y_train_prob))

    y_pred3 = rfc.predict(test_x)


    print("Accuracy Score of Random Forest Classifier : ", accuracy_score(y_pred3,test_y))

    fpr, tpr, thresholds = roc_curve(test_y,y_test_prob)
    thresholds[0] = thresholds[0]-1

    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(fpr,tpr,'black')
    ax.plot(fpr,fpr,'green')
    ax1=ax.twinx()
    ax1.plot(fpr,thresholds)
    ax1.set_ylabel("Thresholds")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    plt.show()

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
# GridSearch(X,Y)
# RandomForest(X,Y,0.25)
# RandomizedSearch(X,Y)
SVM(X,Y,0.25)
# NaiveBayes_Simple(X,Y,0.25)
# RandomForest_K_folds(X,Y)
# SVM_K_folds(X,Y)
# NaiveBayes_K_folds(X,Y)

