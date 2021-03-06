import numpy as np
import pandas as pd
import sklearn.metrics as metrics

import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold, cross_val_score, \
    ShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc, \
    average_precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from scipy.stats import randint as sp_randint
from imblearn.under_sampling import TomekLinks, RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import learning_curve
from collections import Counter


def Open_DataSet():
    df = pd.read_csv("emails.csv")
    X = df.iloc[:,1:3001].values
    Y = df.iloc[:,-1].values
    return X,Y

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):

    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    cmold = cm

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, cmold[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, cmold[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


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
    y_val_stringGlobal = []
    predGlobal = []
    for train_index, test_index in kf.split(X,Y):
        print('-----------------------------------------------------------------')
        print('Fold ' + str(indexFolds))
        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = Y[train_index], Y[test_index]
        mnb = MultinomialNB(alpha=1)
        mnb.fit(X_train,y_train)

        print("Train Results \n")
        y_train_pred  = mnb.predict(X_train)
        y_train_prob = mnb.predict_proba(X_train)[:,1]

        print("Confusion Matrix for Train : \n", confusion_matrix(y_train, y_train_pred))
        print("Accuracy Score for Train : ", accuracy_score(y_train, y_train_pred))
        print("ROC AUC for Train : ", roc_auc_score(y_train, y_train_prob))

        print("+"*50)
        print("Test Results \n")
        y_test_pred  = mnb.predict(X_val)
        y_test_prob = mnb.predict_proba(X_val)[:,1]

        print("Confusion Matrix for Test : \n", confusion_matrix(y_val, y_test_pred))
        print("Accuracy Score for Test : ", accuracy_score(y_val, y_test_pred))
        print("ROC AUC for Test : ", roc_auc_score(y_val, y_test_prob))

        y_val_stringGlobal.extend(y_test_pred)
        predGlobal.extend(y_val)


        y_pred1 = mnb.predict(X_val)
        print("Score: ", accuracy_score(y_pred1,y_val))
        cv_scores.append(accuracy_score(y_pred1,y_val))
        indexFolds = indexFolds + 1

    print('Estimated Accuracy %.3f (%.3f)' % (np.mean(cv_scores), np.std(cv_scores)))

    accuracy = metrics.accuracy_score(predGlobal, y_val_stringGlobal)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = metrics.precision_score(predGlobal, y_val_stringGlobal, average='macro')
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = metrics.recall_score(predGlobal, y_val_stringGlobal, average='macro')
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = metrics.f1_score(predGlobal, y_val_stringGlobal, average='macro')
    print('F1 score: %f' % f1)


    print("Confusion Matrix for Test : \n", confusion_matrix(predGlobal,y_val_stringGlobal))
    Style = ['Spam', 'Not Spam']
    plot_confusion_matrix(confusion_matrix(predGlobal,y_val_stringGlobal),Style,"Classificação Naive Bayes","Blues")


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
    y_val_stringGlobal = []
    predGlobal = []
    for train_index, test_index in kf.split(X,Y):
        print('-----------------------------------------------------------------')
        print('Fold ' + str(indexFolds))
        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = Y[train_index], Y[test_index]
        # svc = SVC(C=1.0,kernel='rbf',gamma='auto')
        svc = SVC(C=100,kernel='rbf',gamma=0.0001,probability=True)
        svc.fit(X_train,y_train)

        print("Train Results \n")
        y_train_pred  = svc.predict(X_train)
        y_train_prob = svc.predict_proba(X_train)[:,1]

        print("Confusion Matrix for Train : \n", confusion_matrix(y_train, y_train_pred))
        print("Accuracy Score for Train : ", accuracy_score(y_train, y_train_pred))
        print("ROC AUC for Train : ", roc_auc_score(y_train, y_train_prob))

        print("+"*50)
        print("Test Results \n")
        y_test_pred  = svc.predict(X_val)
        y_test_prob = svc.predict_proba(X_val)[:,1]

        print("Confusion Matrix for Test : \n", confusion_matrix(y_val, y_test_pred))
        print("Accuracy Score for Test : ", accuracy_score(y_val, y_test_pred))
        print("ROC AUC for Test : ", roc_auc_score(y_val, y_test_prob))

        y_val_stringGlobal.extend(y_test_pred)
        predGlobal.extend(y_val)


        y_pred2 = svc.predict(X_val)
        print("Score: ", accuracy_score(y_pred2,y_val))
        cv_scores.append(accuracy_score(y_pred2,y_val))
        indexFolds = indexFolds + 1

    print('Estimated Accuracy %.3f (%.3f)' % (np.mean(cv_scores), np.std(cv_scores)))

    accuracy = metrics.accuracy_score(predGlobal, y_val_stringGlobal)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = metrics.precision_score(predGlobal, y_val_stringGlobal, average='macro')
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = metrics.recall_score(predGlobal, y_val_stringGlobal, average='macro')
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = metrics.f1_score(predGlobal, y_val_stringGlobal, average='macro')
    print('F1 score: %f' % f1)


    print("Confusion Matrix for Test : \n", confusion_matrix(predGlobal,y_val_stringGlobal))
    Style = ['Spam', 'Not Spam']
    plot_confusion_matrix(confusion_matrix(predGlobal,y_val_stringGlobal),Style,"Classificação SVM","Blues")

def RandomizedSearch(X,Y):

    rfc = RandomForestClassifier(random_state=4)

    params = {'n_estimators': sp_randint(50,400),
              'criterion':['gini','entropy']}

    tuned_parameters = [{'n_estimators': [50,60,70,80,90,100,110,120,130,140],
                         'max_features': [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],
                         'max_depth': [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],
                         'min_samples_split': [2,3,4,5,6,10,12,13,15,20,25],
                         'min_samples_leaf': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,20],
                         'criterion':['gini']}]

    rsearch = RandomizedSearchCV(rfc,
                                param_distributions=params,
                                n_iter=50,
                                cv=3,
                                return_train_score = True,
                                scoring='roc_auc',
                                n_jobs=None,
                                random_state=5)


    # clf = GridSearchCV(rfc, tuned_parameters,
    #                    scoring='roc_auc')

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
    rfc = RandomForestClassifier(n_estimators=100,criterion='gini')
    # rfc = RandomForestClassifier(criterion='entropy', max_depth=8,max_features=13,min_samples_leaf=6,min_samples_split=11,n_estimators=230,random_state=4)
    rfc.fit(train_x,train_y)

    print("Train Results \n")
    y_train_pred  = rfc.predict(train_x)
    y_train_prob = rfc.predict_proba(train_x)[:,1]

    print("Confusion Matrix for Train : \n", confusion_matrix(train_y, y_train_pred))
    print("Accuracy Score for Train : ", accuracy_score(train_y, y_train_pred))
    print("ROC AUC for Test : ", roc_auc_score(train_y, y_train_prob))

    print("+"*50)
    print("Test Results \n")
    y_test_pred  = rfc.predict(test_x)
    y_test_prob = rfc.predict_proba(test_x)[:,1]

    print("Confusion Matrix for Test : \n", confusion_matrix(test_y, y_test_pred))
    print("Accuracy Score for Test : ", accuracy_score(test_y, y_test_pred))
    print("ROC AUC for Test : ", roc_auc_score(test_y, y_test_prob))

    accuracy = metrics.accuracy_score(test_y, y_test_pred)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = metrics.precision_score(test_y, y_test_pred, average='macro')
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = metrics.recall_score(test_y, y_test_pred, average='macro')
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = metrics.f1_score(test_y, y_test_pred, average='macro')
    print('F1 score: %f' % f1)


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
    y_val_stringGlobal = []
    predGlobal = []
    for train_index, test_index in kf.split(X,Y):
        print('-----------------------------------------------------------------')
        print('Fold ' + str(indexFolds))
        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = Y[train_index], Y[test_index]
        rfc = RandomForestClassifier(n_estimators=100,criterion='gini')
        rfc.fit(X_train,y_train)

        print("Train Results \n")
        y_train_pred  = rfc.predict(X_train)
        y_train_prob = rfc.predict_proba(X_train)[:,1]

        print("Confusion Matrix for Train : \n", confusion_matrix(y_train, y_train_pred))
        print("Accuracy Score for Train : ", accuracy_score(y_train, y_train_pred))
        print("ROC AUC for Train : ", roc_auc_score(y_train, y_train_prob))

        print("+"*50)
        print("Test Results \n")
        y_test_pred  = rfc.predict(X_val)
        y_test_prob = rfc.predict_proba(X_val)[:,1]

        print("Confusion Matrix for Test : \n", confusion_matrix(y_val, y_test_pred))
        print("Accuracy Score for Test : ", accuracy_score(y_val, y_test_pred))
        print("ROC AUC for Test : ", roc_auc_score(y_val, y_test_prob))

        y_val_stringGlobal.extend(y_test_pred)
        predGlobal.extend(y_val)

        y_pred3 = rfc.predict(X_val)
        print("Score: ", accuracy_score(y_pred3,y_val))
        cv_scores.append(accuracy_score(y_pred3,y_val))

        indexFolds = indexFolds + 1

    print('Estimated Accuracy %.3f (%.3f)' % (np.mean(cv_scores), np.std(cv_scores)))

    accuracy = metrics.accuracy_score(predGlobal, y_val_stringGlobal)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = metrics.precision_score(predGlobal, y_val_stringGlobal, average='micro')
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = metrics.recall_score(predGlobal, y_val_stringGlobal, average='micro')
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = metrics.f1_score(predGlobal, y_val_stringGlobal, average='micro')
    print('F1 score: %f' % f1)


    print("Confusion Matrix for Test : \n", confusion_matrix(predGlobal,y_val_stringGlobal))
    Style = ['Spam', 'Not Spam']
    plot_confusion_matrix(confusion_matrix(predGlobal,y_val_stringGlobal),Style,"Classificação Random Forest","Blues")

def get_stacking():
    level0 = list()
    level0.append(('MNB', MultinomialNB(alpha=1)))
    level0.append(('RFC', RandomForestClassifier(n_estimators=100,criterion='gini')))
    level0.append(('SVM', SVC(C=100,kernel='rbf',gamma=0.0001)))
    level1 = LogisticRegression()
    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
    return model

def evaluate_model(model, X, Y, name):
    cv_scores = list()
    roc_scores = list()
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=10)
    # kf = ShuffleSplit(n_splits=10, test_size=0.2, random_state=10)

    title = "Learning Curves " + str(name)
    estimator = model
    plot_learning_curve(name, estimator, title, X, Y, ylim=(0.7, 1.01),
                        cv=kf, n_jobs=4)
    plt.show()

    kf.get_n_splits(X)
    y_real = []
    y_proba = []
    indexFolds = 1
    y_val_stringGlobal = []
    predGlobal = []
    for train_index, test_index in kf.split(X,Y):
        # print('-----------------------------------------------------------------')
        # print('Fold ' + str(indexFolds))

        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = Y[train_index], Y[test_index]
        model.fit(X_train,y_train)

        y_test_pred  = model.predict(X_val)
        y_test_prob = model.predict_proba(X_val)[:,1]

        precision, recall, thresholds = precision_recall_curve(y_val, y_test_prob)
        auc_precision_recall = auc(recall, precision)

        plt.plot(recall, precision, lw=1, alpha=0.5, label='PR fold %d (AUC = %0.2f)' % (indexFolds, auc_precision_recall))

        y_real.append(y_val)
        y_proba.append(y_test_prob)

        roc_scores.append(roc_auc_score(y_val, y_test_prob))
        y_val_stringGlobal.extend(y_test_pred)
        predGlobal.extend(y_val)

        y_pred3 = model.predict(X_val)
        cv_scores.append(metrics.f1_score(y_pred3,y_val))

        indexFolds = indexFolds + 1

    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)

    precision, recall, _ = precision_recall_curve(y_real, y_proba)

    plt.plot(recall, precision, color='b',
             label=r'Precision-Recall (AUC = %0.2f)' % (average_precision_score(y_real, y_proba)),
             lw=2, alpha=.8)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(name)
    plt.legend(loc="lower left")
    plt.show()
    plt.show()

    Style = ['Not Spam', 'Spam']
    plot_confusion_matrix(confusion_matrix(predGlobal,y_val_stringGlobal),Style,"Classificação " + name,"Blues")

    print("+"*50)
    print(name)

    accuracy = metrics.accuracy_score(predGlobal, y_val_stringGlobal)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = metrics.precision_score(predGlobal, y_val_stringGlobal, average='micro')
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = metrics.recall_score(predGlobal, y_val_stringGlobal, average='micro')
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = metrics.f1_score(predGlobal, y_val_stringGlobal, average='micro')
    print('F1 score: %f' % f1)

    return cv_scores, roc_scores

def get_models():
    models = dict()
    models['MNB'] = MultinomialNB(alpha=1)
    models['RFC'] = RandomForestClassifier(n_estimators=339,criterion='gini')
    models['SVM'] = SVC(C=100,kernel='rbf',gamma=0.0001,probability=True)
    models['Stacking'] = get_stacking()
    return models

def Stacking(X,Y):
    models = get_models()
    resultsf1, names = list(), list()
    for name, model in models.items():
        cv_scores,roc_scores = evaluate_model(model, X, Y, name)
        resultsf1.append(cv_scores)
        names.append(name)

    pyplot.boxplot(resultsf1, labels=names, showmeans=True)
    pyplot.show()

def InformationDataSet(Y):
    Spam = 0
    NaoSpam = 0
    for x in Y:
        if(x == 0):
            NaoSpam = NaoSpam + 1
        if(x == 1):
            Spam = Spam + 1

    print("Spam:" + str(Spam))
    print("Nao Spam:" + str(NaoSpam))

def OverAndUnderSampling(X,Y):
    over = SMOTE(sampling_strategy=0.5)
    under = RandomUnderSampler(sampling_strategy=0.8)

    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)

    x, y = pipeline.fit_resample(X, Y)

    return x,y


def plot_learning_curve(name, estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model " + str(name))

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model " + str(name))

    return plt

X,Y = Open_DataSet()
# RandomizedSearch(X,Y)
# InformationDataSet(Y)
# X,Y = OverAndUnderSampling(X,Y)
# InformationDataSet(Y)
# GridSearch(X,Y)
# RandomForest(X,Y,0.25)
# RandomizedSearch(X,Y)
# SVM(X,Y,0.25)
# NaiveBayes_Simple(X,Y,0.25)
# RandomForest_K_folds(X,Y)
# SVM_K_folds(X,Y)
# NaiveBayes_K_folds(X,Y)
Stacking(X,Y)
