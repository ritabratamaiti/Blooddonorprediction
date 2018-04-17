import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import KFold
from sklearn import tree
from tpot import TPOTClassifier


X = []
Y = []
list_clf = []

df = pd.read_csv('dataset.csv', na_values = {'?'})
df = df.values


X = df[:, :(df.shape[1]-1)]
Y = df[:, df.shape[1]-1]

tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2)
tpot.fit(X, Y)

#ignore this commented block, it's present only for a (now) obsolete test
'''
kf = KFold(n_splits=3)
kf.get_n_splits(X)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    clf.fit(X_test, Y_test)  
    print(clf.score(X_train, Y_train))  

'''
clf_tree = tree.DecisionTreeClassifier()
clf_svm = SVC()
clf_perceptron = Perceptron()
clf_KNN = KNeighborsClassifier()
clf_nb = BernoulliNB()

list_clf = [clf_tree, clf_svm, clf_perceptron, clf_KNN, clf_nb, tpot.fitted_pipeline_]

kf = KFold(n_splits=5)
kf.get_n_splits(X)
c = 1
for clfs in list_clf:
    print(c)
    c += 1
    a = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        clfs.fit(X_test, Y_test)  
        print(clfs.score(X_train, Y_train)) 
        a += clfs.score(X_train, Y_train)
    a = a/5
    print("Average=",a,"\n")
    print(clfs,"\n")
