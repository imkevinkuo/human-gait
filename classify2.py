import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import sys
import math
import os.path
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, confusion_matrix

def plot_precision_vs_recall(precisions, recalls):
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])
    
def plot_subject(img):
    plt.imshow(img, cmap = matplotlib.cm.binary)
    plt.axis("off")
    plt.show()

def plot(x,y):
    plt.plot(x, y, 'bo', x, y, 'k')
    plt.show()

def import_data():
    data = []
    target = []
    d = os.getcwd()
    d = os.path.join(d, "TreadmillDatasetA_avg")
    subjects = os.listdir(d)
    for subject in subjects:
        d2 = os.path.join(d, subject)
        seqs = os.listdir(d2)
        for seq in seqs: # probe and gallery
            d3 = os.path.join(d2, seq)
            file_names = os.listdir(d3)
            for file_name in file_names:
                file_path = os.path.join(d3, file_name)
                img = cv2.imread(file_path, 0)
                img = cv2.resize(img, (22,32)).flatten()
                data.append(np.uint16(img))
                target.append(int(subject))
    return np.array(data), np.array(target)

def even_sets(X,y):
    X_train, X_test, y_train, y_test = [],[],[],[]
    for i in range(len(y)):
        if y_test.count(y[1]) < 2:
            y_test.append(y[i])
            X_test.append(X[i])
        else:
            y_train.append(y[i])
            X_train.append(X[i])
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

def dist_classify(X_train,y_train,X_test,y_test):
    for i in range(len(X_test)):
        dists = [np.linalg.norm(X_train[j] - X_test[i]) for j in range(len(X_train))]
        m = min(dists)
        mi = dists.index(m)
        print(y_test[i] == y[mi], y_test[i], y[mi], m)
        
def sgd_single(X_train,y_train,X_test,y_test): # binary SGD
    from sklearn.linear_model import SGDClassifier
    tc = 148 # testing class
    y_train_x = (y_train == tc)
    sgd_clf = SGDClassifier(max_iter=5, tol=-np.infty, random_state=42)
    sgd_clf.fit(X_train, y_train_x)
    cvs = cross_val_score(sgd_clf, X_train, y_train_x, cv=4, scoring="accuracy")

    print(cvs)
    y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_x, cv=4)
    cm = confusion_matrix(y_train_x, y_train_pred)
    print(cm)
        
def sgd_classify(X_train,y_train,X_test,y_test,i): # multiclass SGD, OnevAll
    from sklearn.linear_model import SGDClassifier
    sgd_clf = SGDClassifier(max_iter=i, tol=-np.infty, random_state=42)
    sgd_clf.fit(X_train, y_train)
    cvs = cross_val_score(sgd_clf, X_train, y_train, cv=4, scoring="accuracy")

    print(cvs)
    y_train_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=4)
##    cm = confusion_matrix(y_train, y_train_pred)
##    for row in cm:
##        print([x for x in row])
    return val_test(sgd_clf, X_test, y_test)

def sgd_classify_ovo(X_train,y_train,X_test,y_test,i): # multiclass SGD, OnevOne
    from sklearn.linear_model import SGDClassifier
    from sklearn.multiclass import OneVsOneClassifier
    sgd_clf = OneVsOneClassifier(SGDClassifier(max_iter=i, tol=-np.infty, random_state=42))

    sgd_clf.fit(X_train, y_train)
    cvs = cross_val_score(sgd_clf, X_train, y_train, cv=4, scoring="accuracy")

    print(cvs)
    y_train_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=4)
##    cm = confusion_matrix(y_train, y_train_pred)
##    for row in cm:
##        print([x for x in row])
    return val_test(sgd_clf, X_test, y_test)

# Let's try taking problematic classes out of OvA predictor
# and use OvO specifically for those classes


def rf_classify(X_train,y_train,X_test,y_test, n): # random forest
    from sklearn.ensemble import RandomForestClassifier
    rf_clf = RandomForestClassifier(n_estimators=n, random_state=42)
    rf_clf.fit(X_train, y_train)
    cvs = cross_val_score(rf_clf, X_train, y_train, cv=4, scoring="accuracy")

    print(cvs)
    y_train_pred = cross_val_predict(rf_clf, X_train, y_train, cv=4)
##    cm = confusion_matrix(y_train, y_train_pred)
##    print(cm)
    return val_test(rf_clf, X_test, y_test)

def val_test(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    n_correct = sum(y_pred == y_test)
    return n_correct / len(y_pred)
  
## 34 subjects, 716 images (~21 per subject), 22*32 = 704 pixels/features
X,y = import_data()
np.random.seed(0)
shuffle = np.random.permutation(716)
X,y = X[shuffle], y[shuffle]

##acc = []
##for i in range(1,21):
##    a = sgd_classify(*even_sets(X,y), i)
##    a = sgd_classify_ovo(*even_sets(X,y),i)
##    a = sgd_single(*even_sets(X,y))
##    a = rf_classify(*even_sets(X,y), i)
##    acc.append(a)
##plot([i for i in range(1,len(acc)+1)], acc)
