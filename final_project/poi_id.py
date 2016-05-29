#!/usr/bin/python

import sys
import pickle
import pprint
sys.path.append("../tools/")  # noqa
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np


# Options for feature selction
FS_OPTIONS = ['SelectKBest', 'tree']
FEATURE_SELECTOR = FS_OPTIONS[0]
NUM_FEATURES = 10

PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\
\tRecall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\t\
F2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\t\
True positives: {:4d}\tFalse positives: {:4d}\
\tFalse negatives: {:4d}\tTrue negatives: {:4d}"


def getFeatureList(data):
    """Returns a list of the feature names from the dataset
    args:
        data: dictionary containing Enron data
    """
    featureList = ['poi']
    for key, value in data["SKILLING JEFFREY K"].iteritems():
        if (key != 'poi' and key != 'email_address'):
            featureList.append(key)
    return featureList


def addNewFeatures(features_list, my_dataset):
    """Add new features to existing dataset
    args:
        features_list: list containing existing features
        my_dataset: dictionary containing Enron data
    """
    total_messages = ['from_messages', 'to_messages']
    total_messages_with_poi = ['from_poi_to_this_person',
                               'from_this_person_to_poi',
                               'shared_receipt_with_poi']
    excerised_stock_ratio = ['total_stock_value', 'exercised_stock_options']
    loan_ratio = ['total_payments', 'loan_advances']
    for rec, value in my_dataset.items():
        # create empty elements in dictionary
        my_dataset[rec]['total_messages'] = 0
        my_dataset[rec]['total_messages_with_poi'] = 0
        my_dataset[rec]['message_shared_fraction'] = 0
        my_dataset[rec]['excerised_stock_ratio'] = 0
        my_dataset[rec]['loan_ratio'] = 0
        validStock = True
        validLoan = True
        for key in total_messages:
            if value[key] != "NaN":
                my_dataset[rec]['total_messages'] += value[key]
        for key in total_messages_with_poi:
            if value[key] != "NaN":
                my_dataset[rec]['total_messages_with_poi'] += value[key]
        if my_dataset[rec]['total_messages'] > 0:
            my_dataset[rec]['message_shared_fraction'] = \
                float(my_dataset[rec]['total_messages_with_poi']) / \
                my_dataset[rec]['total_messages']

        for key in excerised_stock_ratio:
            if value[key] == "NaN":
                validStock = False
                break
        for key in loan_ratio:
            if value[key] == "NaN":
                validLoan = False
                break
        if validStock and my_dataset[rec]['total_stock_value'] != 0:
            my_dataset[rec]['excerised_stock_ratio'] = \
                float(my_dataset[rec]['exercised_stock_options']) / \
                my_dataset[rec]['total_stock_value']
        if validLoan and my_dataset[rec]['total_payments'] != 0:
            my_dataset[rec]['loan_ratio'] =  \
                float(my_dataset[rec]['loan_advances']) / \
                my_dataset[rec]['total_payments']
    return getFeatureList(my_dataset), my_dataset


def tuneKNeighbour():
    """Prints the best params for the KNeighborsClassifier based on the
    results of the GridSearchCV
    """
    score_metric = 'precision'

    params = {'n_neighbors': range(1, 11), 'weights': ['uniform', 'distance'],
              'p': [1, 2], 'leaf_size': [1, 5, 10, 20, 30, 40, 50, 60],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

    classifier = KNeighborsClassifier()
    search = GridSearchCV(estimator=classifier, param_grid=params,
                          scoring=score_metric, n_jobs=1, refit=True, cv=10)
    search.fit(features, labels)

    print "Best parameters: ", search.best_params_
    print "Best Score: ", score_metric, search.best_score_
    print "Number of tested models: %i" % np.prod(
        [len(params[element]) for element in params])


def tuneDecisionTree():
    """Prints the best params for the DecisionTreeClassifier based on the
    results of the GridSearchCV
    """
    params = {"max_depth": range(1, 11), "max_features": [1, 5, 10],
              "min_samples_split": range(1, 11),
              "min_samples_leaf": range(1, 11),
              "criterion": ["gini", "entropy"]}
    score_metric = 'precision'
    classifier = DecisionTreeClassifier()
    search = GridSearchCV(estimator=classifier, param_grid=params,
                          scoring=score_metric, n_jobs=1, refit=True, cv=10)
    search.fit(features, labels)
    print search.best_params_
    print search.best_score_


def getClassifiers():
    """Returns tuned classifiers
    """
    gnb = GaussianNB()
    # decision tree after tuning with gridsearch
    tree = DecisionTreeClassifier(max_features=10, min_samples_split=4,
                                  criterion='entropy', max_depth=10,
                                  min_samples_leaf=2)

    # kneighbour after tuning with gridsearch
    kneighbour = KNeighborsClassifier(n_neighbors=5, weights='uniform',
                                      leaf_size=1, algorithm='auto', p=1)
    return gnb, tree, kneighbour


def testClassifers(classifiers):
    """Calls the test_classifier from tester.py for each classifier
    args:
        classifiers: list of classifiers to test
    """
    from tester import test_classifier
    print "Classifier Test Results"
    print "==================================="
    for clf in classifiers:
        test_classifier(clf, my_dataset, features_list, 1000)


def getTrainingTestSets(labels, features):
    """ Creates training and test sets based on the StratifiedShuffleSplit
    args:
        labels: list of labels from the data
        features: list of features in the data
    """
    cv = StratifiedShuffleSplit(labels, 1000, random_state=42)
    for train_idx, test_idx in cv:
        features_train = []
        features_test = []
        labels_train = []
        labels_test = []
        for ii in train_idx:
            features_train.append(features[ii])
            labels_train.append(labels[ii])
        for jj in test_idx:
            features_test.append(features[jj])
            labels_test.append(labels[jj])
    return features_train, features_test, labels_train, labels_test


def getBestFeatures(features, labels, showResults=False):
    """ Returns the best features based on the Feature Selection Options
    The features are selected based on the highest score / importance
    args:
        labels: list of labels from the data
        features: list of features in the data
        showResults: boolean set to true to print list of features and scores
    """
    features_train, features_test, labels_train, labels_test = \
        getTrainingTestSets(labels, features)
    revised_feature_list = ['poi']
    if FEATURE_SELECTOR == "tree":
        clf = DecisionTreeClassifier()
        clf = clf.fit(features_train, labels_train)
        importance = clf.feature_importances_
    else:
        k_best = SelectKBest(k=NUM_FEATURES)
        k_best.fit(features_train, labels_train)
        importance = k_best.scores_

    feature_scores = sorted(zip(features_list[1:], importance),
                            key=lambda l: l[1], reverse=True)
    for feature, importance in feature_scores[:NUM_FEATURES]:
        revised_feature_list.append(feature)
    if showResults:
        print "Top features and scores:"
        print "==================================="
        pprint.pprint(feature_scores[:NUM_FEATURES])
    return revised_feature_list


# Retrieve data from pkl file
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


# Task 1: Select what features you'll use.
# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".

features_list = getFeatureList(data_dict)

# Task 2: Remove outliers
data_dict.pop('TOTAL', 0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)


# Task 3: Create new feature(s)
# Store to my_dataset for easy export below.
my_dataset = data_dict
features_list, my_dataset = addNewFeatures(features_list, my_dataset)

# convert dictionary into features and labels
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

# scale features
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

# select best features
features_list = getBestFeatures(features, labels, True)

# Re-split data based on new feature list after getBestFeatures
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

# re-scale
features = scaler.fit_transform(features)

# Get training and test data
features_train, features_test, labels_train, labels_test = \
	getTrainingTestSets(labels, features)


"""
Uncomment to re-run algorithm tuning
tuneKNeighbour()
tuneDecisionTree()
"""

# get classifiers
gnb, tree, kneighbour = getClassifiers()

# test classifiers
testClassifers([gnb, tree, kneighbour])


# select classifier manually based on test results
clf = kneighbour

# Task 6: Dump your classifier, dataset, and features_list so anyone can
# check your results. You do not need to change anything below, but make sure
# that the version of poi_id.py that you submit can be run on its own and
# generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
