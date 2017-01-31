import os
import sys

# Hack solution for import error associated with next import - don't have time to fix it though properly
sys.path.insert(0, os.path.abspath(os.getcwd()))

import src.data.raw.clickstream as clickstream_data


def get_twitter_to_clickstream_dict():
    '''
    Returns a dictionary of {twitter_id:clickstream_id} based off data/processed/twitter_anonid_edge.csv

    ToDO: Consider caching results    
    '''
    twitter_to_clickstream_id = {}
    # load clickstream id to twitter id table
    with open('data/processed/twitter_anonid_edge.csv', 'r') as clickstream_to_twitter_file:
        for line in clickstream_to_twitter_file:
            stripped_line = line.strip()
            if stripped_line:
                split_line = stripped_line.split(',')
                twitter_to_clickstream_id[split_line[0]] = split_line[1]

    return twitter_to_clickstream_id 


def build_labels_features():
    '''
    Retrieves clickstream features

    saves:
       -- clickstream userid

    Stores results in data/processed/clickstream_cluster_labeled_data_features.csv
    '''
    twitter_to_clickstream_id = get_twitter_to_clickstream_dict()
    labeled_clickstream_data = {}

    # grab labels for twitters ids with associated clickstream id
    with open('data/processed/twitter_user_cluster_labels.csv') as twitter_user_cluster_labels_file:
        # skip header of file
        next(twitter_user_cluster_labels_file)
        
        for line in twitter_user_cluster_labels_file:
            stripped_line = line.strip()
            if stripped_line:
                split_line = stripped_line.split(',')
                if split_line[0] in twitter_to_clickstream_id:
                    labeled_clickstream_data[twitter_to_clickstream_id[split_line[0].strip()]] = {'twitter_cluster_label': int(split_line[1].strip())}

    # find the associated features
    # go through part 0
    print "going through part 0 features"
    for clickstream_feature_set in clickstream_data.iget_clickstream_dataentry_part0():
        if clickstream_feature_set[0] in labeled_clickstream_data:
            labeled_clickstream_data[clickstream_feature_set[0]]["part0_features"] = clickstream_feature_set[1:]
    # go through part 1
    print "going though part 1 features"
    for clickstream_feature_set in clickstream_data.iget_clickstream_dataentry_part1():
        if clickstream_feature_set[0] in labeled_clickstream_data:
            labeled_clickstream_data[clickstream_feature_set[0]]["part1_features"] = clickstream_feature_set[1:]
    print "done looking through features"

    # remove clickstream_ids where no clickstream features could be found
    clickstream_ids  = labeled_clickstream_data.keys()
    for clickstream_id in clickstream_ids:
        stored_data = labeled_clickstream_data[clickstream_id]
        if not "part0_features" in stored_data or not "part1_features" in stored_data:
            del labeled_clickstream_data[clickstream_id]

    # save results
    with open ('data/processed/lookalike_model_labels_features.csv', 'w') as lookalike_labels_features_file:
        lookalike_labels_features_file.write("clickstream_id,cluster_label\n")
        for clickstream_id, stored_data in labeled_clickstream_data.iteritems():
            data_str = clickstream_id + ',' + str(stored_data['twitter_cluster_label'])
            for feature in stored_data["part0_features"]:
                data_str += ',' + str(feature)
            for feature in stored_data["part1_features"]:
                data_str += ',' + str(feature)
            data_str += '\n'
            lookalike_labels_features_file.write(data_str)
            

if __name__ == '__main__':
    build_labels_features()
