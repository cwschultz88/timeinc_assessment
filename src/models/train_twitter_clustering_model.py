import csv
import numpy as np
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

def train_final_model(assign_labels_to_twitter_users=True):
    '''
    trains final model using k-means algorithm

    Increases clusters until at least 4 clusters are found with 1000 members

    Prunes out small clusters then and does a final labeling of all the twitter data

    Prints out characteristics qualities

    If assign_labels_to_twitter_users is True, assigns cluster labels to
    all twitter users in twitter features files
    '''
    # get twitter data
    user_info, features_set, features_header = _get_twitter_features()

    # increase number of clusters in k means until found at leasst 4 clusters with 1000 members
    k = 15
    while True:
        model = KMeans(k)
        unpruned_labels = model.fit_predict(features_set)
        number_of_unqiue_labels = len(set(unpruned_labels.tolist()))
        counts_of_each_label = [unpruned_labels.tolist().count(i) for i in xrange(number_of_unqiue_labels)]

        number_of_clusters_with_1000_members = 0
        for label_count in counts_of_each_label:
            if label_count >= 1000:
                number_of_clusters_with_1000_members += 1
        if number_of_clusters_with_1000_members >= 4:
            break

        if k == 100:
            print "**** FAILED -- no k <= 100 yielded 4 clusters of at least 1000 members ****"
            return

        k += 1

    # prune out clusters with less than 1000 members
    model_labels = model.labels_
    pruned_centers = []
    for i,center in enumerate(model.cluster_centers_):
        if len([j for j in model_labels if i == j]) >= 1000:
            pruned_centers.append(center)
    model.cluster_centers_ = np.array(pruned_centers)

    # get new labels
    pruned_labels = model.predict(features_set)
    print len(set(pruned_labels.tolist()))








if __name__ == '__main__':
    train_final_model()