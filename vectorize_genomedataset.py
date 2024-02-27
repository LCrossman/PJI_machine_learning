#!/usr/bin/python

#original script from 2017, supported python 2.7 and sklearn 0.18.1 or sklearn 0.18.2 with numpy ~ 1.13.0

from __future__ import division
import math
import pickle
import cPickle
import numpy as np
from time import time
import scipy.sparse as sp
import matplotlib.pyplot as plt
from itertools import *
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import Normalizer, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostRegressor, ExtraTreesRegressor, ExtraTreesClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from sklearn.preprocessing import Imputer, Normalizer
from sklearn.feature_selection import SelectPercentile, f_classif, chi2


def plotfeatImportances(importances, indices, std, X):
     plt.figure()
     plt.title("Feature importances")
     plt.bar(range(X.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
     plt.xticks(range(X.shape[1]), indices)
     plt.xlim([-1, X.shape[1]])
     plt.show()


def selectKImportance(model, X, k=5):
     return X[:,model.feature_importances_.argsort()[::-1][:k]]

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    from itertools import product
    import itertools
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def benchmark(clf_class, params, name):
    print("parameters:", params)
    t0 = time()
    clf = MultinomialNB(fit_prior=True, class_prior=None)
    clf.fit(features_train, labels_train)
#    clf = OneVsRestClassifier(clf_class(**params).fit(features_train, labels_train))
    print("done in %fs" % (time() - t0))

    if hasattr(clf, 'coef_'):
        print("Percentage of non zeros coef: %f"
              % (np.mean(clf.coef_ != 0) * 100))
    print("Predicting the outcomes of the testing set")
    t0 = time()
    pred = clf.predict(features_test)
    print("done in %fs" % (time() - t0))

#    print("Classification report on test set for classifier:")
#    print(classification_report(features_test, pred, target_names=labels_test))

    cm = confusion_matrix(features_test, pred)
    print("Confusion matrix:")
    print(cm)
    print(cm.diagonal()/cmat.sum(axis=1))
    # Show confusion matrix
    plt.matshow(cm)
    plt.title('Confusion matrix of the %s classifier' % name)
    plt.colorbar()


def doPCA(data):
    from sklearn.decomposition import TruncatedSVD
    pca = PCA(n_components=2)
    pca.fit(data)
    return pca

def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical

def parseOutText(f):
    f.seek(0)
    all_text = f.read()
    words = ""
    if len(all_text) > 1:
        text_string = all_text.rstrip()
        elements = text_string.split()
        words = elements
    return words
    
def show_most_informative_features(vectorizer, clf, n=20):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print "\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2)

def preprocess():
   from_auth = []
   word_data = []
   
   d = {}
   from_1 = open("proc1_20_py.fofn", 'r')
   from_2 = open("proc2_20_py.fofn", 'r')
   from_3 = open("proc3_20_py.fofn", 'r')
   from_4 = open("proc4_20_py.fofn", 'r')
   for name, from_dat in [('1', from_1), ('2', from_2), ('3', from_3), ('4', from_4)]:
       for path in from_dat:
           wortg = path.rstrip()
           print(wortg)
           w = open(wortg, 'r')
           text = parseOutText(w)
           word_data.append(text)
           if name == "1":
              from_auth.append('1')
           elif name == "2":
              from_auth.append('2')
           elif name == "3":
              from_auth.append('3')
           elif name == "4":
              from_auth.append('4')
           else:
              print "seems to be an issue"
   

   print "train_test_splitting..."
   features_train, features_test, labels_train, labels_test = train_test_split(word_data, from_auth, test_size=0.25, random_state=42)
   print "splitting finished..."
   print "vectorizing..."
   vectorizer = TfidfVectorizer(sublinear_tf=True, use_idf=True,smooth_idf=True, tokenizer=lambda doc: doc, lowercase=False)
   print "vectorizing finished..."
   print "transforming..."
   print type(features_test)
   features_train_transformed = vectorizer.fit_transform(features_train)
   print "training transform finished..."
   print "test features doing..."
   features_test_transformed = vectorizer.transform(features_test)
   print "test features done"
   #print "normalizing"
   #scaler = Normalizer(copy=False).fit(features_train_transformed)
   #features_train_transformed = scaler.transform(features_train_transformed)
   #features_test_transformed = Normalizer(copy=False).transform(features_test_transformed)
   print "done normalizing"
   print "dealing with features_names"
   feature_names = vectorizer.get_feature_names()
   print "selecting percentile"
   selector = SelectPercentile(score_func=chi2, percentile=33)
   #SelectPercentile(f_classif, percentile=10)
   print "selecting done, fitting"
   selector.fit(features_train_transformed, labels_train)
   print "dealing with feature_names"
   if feature_names:
       feature_names = [feature_names[i] for i in selector.get_support(indices=True)]
       feature_names = np.asarray(feature_names)
   print feature_names.shape
   print "selected features", feature_names
   print "fitting done"
   print "creating the array"
   features_train_transformed = selector.transform(features_train_transformed)
   assert sp.issparse(features_train_transformed)
   features_test_transformed = selector.transform(features_test_transformed)
   #scaler.fit(features_train_transformed)
   assert sp.issparse(features_test_transformed)
   print(features_test_transformed)
   print("labels train", labels_train)
   print("labels test", labels_test)
   #features_train_transformed = scaler.transform(features_train_transformed)
   #assert sp.issparse(features_train_transformed)
   #scaler.transform(features_test_transformed)
   #assert sp.issparse(features_test_transformed)
   print "returning from function"
   #show_most_informative_features(vectorizer, selector, n=50)
   return feature_names, vectorizer, features_train_transformed, features_test_transformed, labels_train, labels_test
   words.close()
   auths.close()

def show_top25(classifier, vectorizer, categories):
     feature_names = np.asarray(vectorizer.get_feature_names())
     for i, category in enumerate(categories):
         top25 = np.argsort(classifier.coef_[i])[-25:]
         print("%s: %s" % (category, " ".join(feature_names[top25])))

feature_names, vectorizer, features_train, features_test, labels_train, labels_test = preprocess()

print "preprocessing concluded" #, len(features_train), len(features_test)
print feature_names
feat = feature_names
d = {}
d = {'1':1,'2':1,'3':100,'4':50}
t0 = time()
from sklearn import tree
#from sklearn.preprocessing import MultiLabelBinarizer # to fit MultiLabelBinarizer for Regression 
#mlb = MultiLabelBinarizer()
#mlb_train = mlb.fit_transform(labels_train)
#mlb_test = mlb.fit_transform(labels_test)
#clf = ExtraTreesRegressor(n_estimators=1000, n_jobs=-1) # to fit the biofilm categories as a sliding numerical scale instead
clf = ExtraTreesClassifier(n_estimators=1500) #gave the best results on accuracy, better than random forests
#clf = DecisionTreeClassifier(max_depth=15) # to fit a single decision tree
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0,3),"s"
class_names = ['1','2','3','4'] # alter to integer to fit the regressor
t1= time()
predicted = clf.predict(features_test)
print "predicting time:", round(time()-t1,3),"s"

###uncomment from this section to get plots of top feature importances
#show_top25(clf, vectorizer, class_names)
#importances = clf.feature_importances_
#std = np.std(clf.feature_importances_,axis=0)
#indices = np.argsort(importances)[::-1]
#std = np.std([importan for importan in importances])
#importances = selectKImportance(clf,features_train,25)
#importances = clf.feature_importances_
#indices = np.argsort(importances)
#features = features_train.columns
#f,ax = plt.subplots(figsize=(11,9))
#plt.title("Feature ranking", fontsize=20)
#plt.bar(range(importances.shape[0]),importances[indices], color="b", align="center")
#plt.xticks(range(importances.shape[0], indices))
#plt.xlim([-1, importances.shape[0]])
#plt.ylabel("importance", fontsize=18)
#plt.xlabel("index of the feature", fontsize=18)

#####top_ranked_features = sorted(enumerate(clf.feature_importances_),key=lambda x:x[1], reverse=True)
#####print top_ranked_features
#####outfile = open("top_rank.txt", 'w')
#####top_ranked_features_indices = map(list, zip(*top_ranked_features))[0]
#####top_ranked_features_p = map(list, zip(*top_ranked_features))[1]
#####print len(top_ranked_features_indices), "len"
#####for fe in top_ranked_features_indices:
   ##### print feat[fe], top_ranked_features_p[fe]
    #####outfile.write("{}\n".format(feat[fe]))
#    print("%s. feature %d (%f)"%(feat[fe+1], top_ranked_features_indices[fe], importances[top_ranked_features_indices[fe]]))
#plt.figure()
#plt.title("Feature importances")
#plt.barh(range(len(top_ranked_features_indices)), importances[top_ranked_features_indices], color="r", align="center")
#plt.yticks(range(len(top_ranked_features_indices)), top_ranked_features[top_ranked_features_indices])
#plt.xlim([-1, features_train.shape[1]])
#plt.show()
#print feat[top_ranked_features_indices]
#outfile.write("{}\n".format([fe for fe in feat[top_ranked_features_indices]]))
#for feature_pvalue in zip(feat[top_ranked_features_indices],clf.pvalues_[top_ranked_features_indices]):
#    print feature_pvalue

#plt.figure(1)
#plt.title('Feature importances')
#plt.barh(range(len(indices)), feat[indices], color='b', align='center')
#plt.yticks(range(len(indices)), feat[indices])
#plt.xlabel('Relative importance')
#plt.show()
#print "features sorted by their score:"
#print importances

#print("Feature ranking:")
#for f in range(10):
#    print("%d. feature %d (%f)"% (f + 1, indices[f], importances[indices[f]]))

#plt.figure()
#plt.title("Feature importances")
#plt.bar(range(10), importances[indices], color="r", yerr=std[indices], align="center")
#plt.xticks(range(10), indices)
#plt.xlim([-1,10])
#plt.show()
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(predicted, labels_test)
print "accuracy=", accuracy
num_classes=4
#y_categorical = to_categorical(mlb_test, num_classes)
#pred_categorical = to_categorical(predicted, num_classes)
#cnf_matrix = confusion_matrix(pred_categorical.argmax(1), y_categorical.argmax(1))
cnf_matrix = confusion_matrix(predicted, labels_test)
np.set_printoptions(precision=2)
class_names = ['1','2','3','4'] # alter to integer format to fit the regressor
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix')

plt.show()

import pydotplus
#exporting 10 alternative trees from the extratrees
i_tree = 0
for tree_in_forest in clf.estimators_:
   if i_tree < 11:
       with open('tree_' + str(i_tree) + "dot", 'w') as my_file:
           dot_data = tree.export_graphviz(tree_in_forest, out_file=None, feature_names = feature_names, class_names=class_names)
           graph = pydotplus.graph_from_dot_data(dot_data)
           graph.write_pdf("Pfam__1234"+str(i_tree)+".pdf")
           i_tree+=1

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
print("feature ranking:")
for f in xrange(len(importances[indices])):
     print("%d. feature %s (%f)" %(f+1, feature_names[indices[f]], importances[indices[f]]))


with open("iris.dot", 'w') as fa:
    fa = tree.export_graphviz(clf, out_file=fa)

#plotfeatImportances(importances, indices, std, features_train)

print("Testbenching a linear classifier...")
parameters = {
    'loss': 'hinge',
    'penalty': 'l2',
    'n_iter': 50,
    'alpha': 0.00001,
    'fit_intercept': True,
}
