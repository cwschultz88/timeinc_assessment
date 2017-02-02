import numpy as np
import os
import pandas as pd
import pickle
import random
import re
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import sys

# Hack solution for import error associated with next import - don't have time to fix it though properly
sys.path.insert(0, os.path.abspath(os.getcwd()))

import src.data.raw.clickstream as clickstream_data


def _load_social_labeling_models():
    '''
    Load Pickled Social labeling models
    '''
    social_labeling_model_files_directory_contents = os.listdir("models/look_alike_social_labeling_models")
    model_filenames = [filename for filename in social_labeling_model_files_directory_contents if '.p' in filename]
    if len(model_filenames) < 1:
        print "No Model Files Found - run train_social_clustering_label_models() in this module first"
        return
    social_labeling_models = {}
    for model_filename in model_filenames:
        associated_social_label = model_filename.split('_')[2][:-2]
        with open("models/look_alike_social_labeling_models/" + model_filename, 'rb') as model_file:
            social_labeling_models[associated_social_label] = pickle.load(model_file)
    return social_labeling_models


def train_social_clustering_label_models(cluster_labels=set(['0','1','2','3'])):
    '''
    Trains a look alike model for each of cluster labels

    cluster_labels is a list of str labels used to denote the social clustering labels established with clustering twitter data
    '''
    label_features_sets = {label:[] for label in cluster_labels}

    with open("data/processed/lookalike_model_labels_features.csv", 'r') as lookalike_labels_features_file:
        #skip header file
        next(lookalike_labels_features_file)

        for label_features_line in lookalike_labels_features_file:
            split_label_features_line = label_features_line.strip().split(',')
            if len(split_label_features_line) > 1:
                # add features example to associated feature_set for the applicable social cluster label
                label_features_sets[split_label_features_line[1]].append([int(feature) for feature in split_label_features_line[2:]])

    social_clustering_labeling_models = {label:RandomForestClassifier(n_estimators=50, n_jobs=4) for label in cluster_labels}
    for social_cluster_label in cluster_labels:
        print "*** Social Labeling Model " + social_cluster_label + " ***"
        social_label_model = social_clustering_labeling_models[social_cluster_label]
        number_of_positive_examples = len(label_features_sets[social_cluster_label])
        print "Number of Positive Examples: " + str(number_of_positive_examples)

        # get negative examples to train against
        # ToDO: Currently memory inefficient but it will work
        negative_features = []
        for other_cluster_label in cluster_labels ^ set([social_cluster_label]):
            for features_example in label_features_sets[other_cluster_label]:
                negative_features.append(features_example)
        print "Number of Negative Examples: " + str(len(negative_features))
        print "fraction of Positive Examples: " + str(float(number_of_positive_examples)/(len(negative_features) + number_of_positive_examples))
        # Randomize Negative Features
        negative_features = random.sample(negative_features, len(negative_features))

        model_labels = [1 for i in xrange(number_of_positive_examples)] + [0 for i in xrange(len(negative_features))]
        model_features = label_features_sets[social_cluster_label] + negative_features
        model_labels, model_features = zip(*random.sample(zip(model_labels, model_features), len(model_labels)))

        X_train, X_test, y_train, y_test = train_test_split(model_features, model_labels, test_size=0.1)

        print "Fitting Model"
        social_label_model.fit(X_train, y_train)
        y_predicted = social_label_model.predict(X_test)
        y_predicted_prob = social_label_model.predict_proba(X_test)

        print "Fitting Done, 10% validation stats"
        print "accuracy: " + str(accuracy_score(y_test, y_predicted))

        print "saving model"
        with open("models/look_alike_social_labeling_models/social_label_" + social_cluster_label +'.p', 'wb') as model_save_file:
            model_save_file.write(pickle.dumps(social_label_model))
        print ""


def create_gains_chart_info():
    '''
    Stores necessary info to create gain charts in:
        data/processed/clickstream_gain_chart_info.csv

    Assumes only social labels are 0,1,2,3 at the moment
    '''
    social_labeling_models = _load_social_labeling_models()

    df_social_labeled_data = pd.read_csv('data/processed/lookalike_model_labels_features.csv', skiprows=[0])

    # convert labels into binarized labels (one off encoding?)
    # e.x. convert 2 into [0,0,1,0]
    labels = df_social_labeled_data.ix[:,1].tolist()
    binarized_labels = []
    for label in labels:
        if label == 0:
            binarized_labels.append([1, 0, 0, 0])
        elif label == 1:
            binarized_labels.append([0, 1, 0, 0])
        elif label == 2:
            binarized_labels.append([0, 0, 1, 0])
        elif label == 3:
            binarized_labels.append([0, 0, 0, 1])
        else:
            Exception("Only accepts labels 0-3 currently")
    binarized_labels = np.array(binarized_labels)

    features = df_social_labeled_data.ix[:,2:].as_matrix()
    gain_chart_info = df_social_labeled_data.ix[:,0:1].as_matrix()
    del(df_social_labeled_data)
    gain_chart_info = np.append(gain_chart_info, binarized_labels, 1)
    for i in xrange(4):
        gain_chart_info = np.append(gain_chart_info, social_labeling_models[str(i)].predict_proba(features)[:,1:], 1)

    df_results = pd.DataFrame(data=gain_chart_info,index=gain_chart_info[:,0],columns=['clickstream_userid','S0','S1','S2','S3','Model Score S0', 'Model Score S1', 'Model Score S2', 'Model Score S3'])
    df_results.to_csv('data/processed/clickstream_gain_chart_info.csv')


def predict_probabilities_of_social_labels(clickstream_part0_filename="/media/chris/EE06AB7406AB3C89/Documents and Settings/Chris/Downloads/UserSessions_2016_11_15_part0.csv", 
                                           clickstream_part1_filename="/media/chris/EE06AB7406AB3C89/Documents and Settings/Chris/Downloads/UserSessions_2016_11_15_part1.csv", 
                                           downsample_clickstream_to=100000):
    '''
    Predicts the probabilities of all clickstream data using traing social label prediction models

    saves results in:
        data/processed/clickstream_social_label_predictions.csv
    '''
    social_labeling_models = _load_social_labeling_models()

    #select rows to load from clickstream_data
    number_of_clickstream_entries = sum(1 for line in open(clickstream_part0_filename)) #number of records in file
    skip = sorted(random.sample(xrange(0,number_of_clickstream_entries),number_of_clickstream_entries-downsample_clickstream_to))

    df_part0 = pd.read_csv(clickstream_part0_filename, skiprows=skip, header=None)
    df_part1 = pd.read_csv(clickstream_part1_filename, skiprows=skip, header=None)
    df_combined = pd.merge(df_part0, df_part1, on=0, how='inner')
    del df_part0
    del df_part1

    features = df_combined.ix[:,1:].as_matrix()
    clickstream_userids = df_combined.ix[:,0:1].as_matrix()
    del df_combined

    results_df_header = ['clickstream_userid']
    results = clickstream_userids
    for social_label, social_label_model in social_labeling_models.iteritems():
        results_df_header.append('social_label_' + social_label)
        results = np.append(results, social_label_model.predict_proba(features)[:,1:], 1)

    df_results = pd.DataFrame(data=results,index=results[:,0],columns=results_df_header)
    df_results.to_csv('data/processed/clickstream_social_label_predictions.csv')

if __name__ == '__main__':
    # train_social_clustering_label_models()
    # predict_probabilities_of_social_labels()
    create_gains_chart_info()
