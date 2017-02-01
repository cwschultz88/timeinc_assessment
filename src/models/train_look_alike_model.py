import random
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

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

    social_clustering_labeling_models = {label:RandomForestClassifier(n_estimators=300, max_depth=10, max_features=0.4, n_jobs=4) for label in cluster_labels}
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
    
if __name__ == '__main__':
    train_social_clustering_label_models()
