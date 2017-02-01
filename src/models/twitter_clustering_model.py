import csv
import numpy as np
import pickle
from operator import itemgetter
import random
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import time

def _get_twitter_features():
    '''
    Extracts twitter features for clustering
    '''
    with open("data/processed/twitter_data_features.csv") as twitter_features_file:
        csv_reader = csv.reader(twitter_features_file)

        features_header = next(csv_reader)[3:]  # ignore userid, screen_name, name as there are not features

        user_info, features_set = ([], [])
        for data_line in csv_reader:
            features_set.append([int(feature_str) for feature_str in data_line[3:]])
            user_info.append([feature_str for feature_str in data_line[:3]])

        return user_info, features_set, features_header

def _print_out_feature_characteristics_of_cluster(features_set, features_header, labels):
    '''
    Prints out a listing of characteristics of features in each cluster
    '''
    number_of_unqiue_labels = len(set(labels.tolist()))
    sorted_features_by_label = {i: [] for i in xrange(number_of_unqiue_labels)}

    for features, label in zip(features_set, labels):
        sorted_features_by_label[label].append(features)
    cluster_feature_means = [np.array(sorted_features_by_label[i]).mean(0).tolist() for i in xrange(number_of_unqiue_labels)]
    print "Average Feature Value by Clusters: "
    for i, feature_name in enumerate(features_header):
        str_clusters_feature_average_value = ""
        for j in xrange(number_of_unqiue_labels):
            str_clusters_feature_average_value += str(cluster_feature_means[j][i]) + "  "
        print "  --" + feature_name + " - " + str_clusters_feature_average_value
    print ""

def compare_models():
    '''
    Train different clustering models and compare silhouette_scores and training times
    '''
    user_info, features_set, features_header = _get_twitter_features()

    #General Characteristics of Features
    print "Average Value of a Feature of Entire Twitter Data:"
    feature_means = np.array(features_set).mean(0).tolist()
    for feature_name, feature_average_value  in zip(features_header, feature_means):
        print "  --" + feature_name + " - " + str(feature_average_value)
    print ""

    models_to_train = [("k-means 5 clusters", KMeans(n_clusters=5)), ("k-means 10 clusters", KMeans(n_clusters=10)), ("k-means 12 clusters", KMeans(n_clusters=12)), ("k-means 13 clusters", KMeans(n_clusters=13)), ("k-means 14 clusters", KMeans(n_clusters=14)), ("k-means 15 clusters", KMeans(n_clusters=15)),  ("k-means 17 clusters", KMeans(n_clusters=17)),("k-means 20 clusters", KMeans(n_clusters=20)), ("k-means 25 clusters", KMeans(n_clusters=25)), ("EM Clustering 4 Clusters", GaussianMixture(4)), ("DBSCAN", DBSCAN())]
    for model_description, model in models_to_train:
        print "----- " + model_description + " -----"

        print "Starting Fitting"
        beginning_time = time.time()
        model.fit(features_set)
        labels = model.predict(features_set)
        print "Fitting Complete"

        print "Time Needed: " + str(time.time() - beginning_time)

        # need to downsample to be able to fit the silhouette calculation in memory on my laptop
        samples_features_set, sampled_labels = zip(*random.sample(zip(features_set, labels), 10000))
        try:
            print "Silhouette Score: " + str(silhouette_score(np.array(samples_features_set), np.array(sampled_labels)))
        except:
            print "** Could not Calculate Silhouette Score **"
        # number of unique labels
        number_of_unqiue_labels = len(set(labels.tolist()))
        print "Number of Clusters: " + str(number_of_unqiue_labels)

        print "Number of Data points in Each Cluster:"
        for i in xrange(number_of_unqiue_labels):
            print "  -- Cluster " + str(i + 1) + ": " + str(labels.tolist().count(i))

        _print_out_feature_characteristics_of_cluster(features_set, features_header, labels)

def train_final_model(k=125, prune_down_to=4):
    '''
    trains final model using k-means algorithm

    Prunes Down to 4 clusters

    Prints out characteristics qualities

    ToDo: If time, implement https://elki-project.github.io/tutorial/same-size_k_means
    '''
    # get twitter data
    user_info, features_set, features_header = _get_twitter_features()

    print "KMeans Model Fitting with k=" + str(k)
    model = KMeans(k)
    unpruned_labels = model.fit_predict(features_set)
    number_of_unqiue_labels = len(set(unpruned_labels.tolist()))
    counts_of_each_label = [unpruned_labels.tolist().count(i) for i in xrange(number_of_unqiue_labels)]
    counts_and_index_of_each_label = [(i, count) for i, count in enumerate(counts_of_each_label)]
    sorted_counts_and_index_of_each_label = sorted(counts_and_index_of_each_label, key=itemgetter(1), reverse=True)

    # prunt
    print 'prunning down to ' + str(prune_down_to) + ' clusters'
    keep_these_centers = set([sorted_counts_and_index_of_each_label[i][0] for i in xrange(prune_down_to)])
    pruned_centers = []
    for i, center in enumerate(model.cluster_centers_):
        if i in keep_these_centers:
            pruned_centers.append(center)
    model.cluster_centers_ = np.array(pruned_centers)

    # get new labels
    pruned_labels = model.predict(features_set)
    number_of_unqiue_labels = len(set(pruned_labels.tolist()))
    print "Cluster member counts: "
    for i in xrange(number_of_unqiue_labels):
        print "  -- cluster " + str(i + 1) + ": " + str(pruned_labels.tolist().count(i))

    # get feature averages with each cluster to assign meaningful labels
    _print_out_feature_characteristics_of_cluster(features_set, features_header, pruned_labels)

    # save model
    with open("models/twitter_clustering_model.p", "wb") as model_file:
        model_file.write(pickle.dumps(model))

    # save twitter user cluster labels
    with open("data/processed/twitter_user_cluster_labels.csv", 'w') as twitter_user_cluster_file:
        twitter_user_cluster_file.write("Twitter Userid,Cluster Label\n")
        for user, label in zip(user_info, pruned_labels):
            twitter_user_cluster_file.write(str(user[0] + ',' + str(label) + '\n'))

if __name__ == '__main__':
    train_final_model()
