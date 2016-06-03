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
import matplotlib.pyplot as plt

# Options for feature selction
FS_OPTIONS = ['SelectKBest', 'tree']
FEATURE_SELECTOR = FS_OPTIONS[0]


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
        validStock = True
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
        if validStock and my_dataset[rec]['total_stock_value'] != 0:
            my_dataset[rec]['excerised_stock_ratio'] = \
                float(my_dataset[rec]['exercised_stock_options']) / \
                my_dataset[rec]['total_stock_value']
    return getFeatureList(my_dataset), my_dataset


def tuneKNeighbour():
    """Prints the best params for the KNeighborsClassifier based on the
    results of the GridSearchCV
    """
    score_metric = 'precision'

    params = {'n_neighbors': range(1, 11), 'weights': ['uniform', 'distance'],
              'p': [1, 2], 'leaf_size': [1, 5, 9, 10, 20, 30, 40, 50, 60],
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
    tree = DecisionTreeClassifier(max_features=9, min_samples_split=4,
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
    PERF_FORMAT_STRING = "\
    \tAccuracy: {:>0.{display_precision}f}\t\
    Precision: {:>0.{display_precision}f}\t\
    Recall: {:>0.{display_precision}f}\t"
    RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\t\
    True positives: {:4d}\tFalse positives: {:4d}\
    \tFalse negatives: {:4d}\tTrue negatives: {:4d}"
    print "Classifier Test Results"
    print "==================================="
    for clf in classifiers:
        total_predictions, accuracy, precision, recall, true_positives, \
            false_positives, true_negatives, false_negatives, f1, f2 = \
            test_classifier(clf, features, labels)
        print clf
        print PERF_FORMAT_STRING.format(accuracy, precision, recall,
                                        display_precision=5)
        print RESULTS_FORMAT_STRING.format(total_predictions, true_positives,
                                           false_positives, false_negatives,
                                           true_negatives)


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


def scoreNumFeatures(test_feature_list, test_data_set):
    """ function for determining the best number of features to use
    """
    scaler = MinMaxScaler()
    recall_scores = []
    precision_scores = []
    feature_count = []
    f1_scores = []
    PERF_FORMAT_STRING = "\
    Features: {:>0.{display_precision}f}\t\
    Accuracy: {:>0.{display_precision}f}\t\
    Precision: {:>0.{display_precision}f}\t\
    Recall: {:>0.{display_precision}f}\t\
    F1: {:>0.{display_precision}f}\t\
    "

    gnb, tree, kneighbour = getClassifiers()
    clf = kneighbour
    for x in range(1, len(test_feature_list)):
        test_data = featureFormat(test_data_set, test_feature_list,
                                  sort_keys=True)
        test_labels, test_features = targetFeatureSplit(test_data)
        test_features = scaler.fit_transform(test_features)
        best_features = getBestFeatures(test_features, test_labels, x, False)
        # Resplit data using best feature list
        test_data = featureFormat(test_data_set, best_features,
                                  sort_keys=True)
        test_labels, test_features = targetFeatureSplit(test_data)
        test_features = scaler.fit_transform(test_features)
        total_predictions, accuracy, precision, recall, true_positives, \
            false_positives, true_negatives, false_negatives, f1, f2 = \
            test_classifier(clf, test_features, test_labels)
        print PERF_FORMAT_STRING.format(x, accuracy, precision, recall, f1,
                                        display_precision=5)
        recall_scores.append(recall)
        precision_scores.append(precision)
        f1_scores.append(f1)
        feature_count.append(x)

    plt.plot(feature_count, recall_scores, marker='o', label="Recall")
    plt.plot(feature_count, precision_scores, marker='o', label="Precision")
    plt.plot(feature_count, f1_scores, marker='o', label="F1")
    plt.legend()
    plt.show()


def test_classifier(clf, features, labels):
    cv = StratifiedShuffleSplit(labels, 1000, random_state=42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
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

        # fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        total_predictions = np.sum([true_negatives, false_negatives,
                                    false_positives, true_positives])
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives +
                                   false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        return total_predictions, accuracy, precision, recall,\
            true_positives, false_positives, true_negatives, \
            false_negatives, f1, f2
    except:
        print "Got a divide by zero when trying out:", clf
        print "Precision or recall may be undefined due to a lack of \
        true positive predicitons."


def getBestFeatures(features, labels, num_features=10, showResults=False):
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
        k_best = SelectKBest(k=num_features)
        k_best.fit(features_train, labels_train)
        importance = k_best.scores_

    feature_scores = sorted(zip(features_list[1:], importance),
                            key=lambda l: l[1], reverse=True)
    for feature, importance in feature_scores[:num_features]:
        revised_feature_list.append(feature)
    if showResults:
        print "Top features and scores:"
        print "==================================="
        pprint.pprint(feature_scores[:num_features])
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

"""
# Uncomment to produce plot which compares evaluates the number of features
# used

scoreNumFeatures(features_list, my_dataset)
"""

# convert dictionary into features and labels
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

# scale features
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

# select best features
features_list = getBestFeatures(features, labels, 10, True)

# Re-split data based on new feature list after getBestFeatures
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

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
