import os
import re
import sys
import json

# Hack solution for import error associated with next import - don't have time to fix it though properly
sys.path.insert(0, os.path.abspath(os.getcwd()))

import src.data.raw.twitter as raw_twitter_data

# Cached Info
brands_twitter_info_table = {}

def extract_brand(twitter_json_object):
    '''
    Given a twitter json obect, extracts the associated Time Inc brand. Return empty string if cannot find the brand

    Assumes only one brand in tweet currently

    If not a valid twitter json object, i.e. expected fields, will yield an exception currently
    '''
    # read in Time Inc. brands twitter info from file if not cached
    if len(brands_twitter_info_table) == 0:
        with open("data/misc/time_inc_brands_twitter_info.csv", "rb") as brands_datafile:
            # skip header line
            next(brands_datafile)

            for line in brands_datafile:
                split_line = line.split(',')

                # skip line if empty
                if len(split_line) == 0:
                    continue

                # cache results
                brands_twitter_info_table[split_line[0]] = (split_line[1].strip(), split_line[2].strip())

    # First Approach - Check for Brand in Entities User Mentions
    user_mentions_data = twitter_json_object['entities']['user_mentions']
    for user_mention in user_mentions_data:
        if 'name' in user_mention:
            if user_mention['name'] in brands_twitter_info_table:
                return user_mention['name']

    # Second Approach - Check the Entities urls for references to a brand
    urls_data = twitter_json_object['entities']['urls']
    urls_to_check_for_brand = []
    # Add 'unwound' and 'expanded' urls in twitter data to list of urls to check for brand references
    for url_data in urls_data:
        if 'unwound' in url_data and url_data["unwound"]['url']:
            urls_to_check_for_brand.append(url_data['unwound']['url'])
        if "expanded_url" in url_data:
            urls_to_check_for_brand.append(url_data["expanded_url"])
    # Check for brand references
    for url_to_check in urls_to_check_for_brand:
        for brand in brands_twitter_info_table:
            brand_url_hint = brands_twitter_info_table[brand][1]
            if brand_url_hint in url_to_check:
                return brand

    # Final Approach - search raw string for brand url, company names, etc. brute force search!
    json_text = json.dumps(twitter_json_object)
    for brand in brands_twitter_info_table:
            brand_url_hint = brands_twitter_info_table[brand][1].strip()
            brand_at_name = brands_twitter_info_table[brand][0]
            # use lower case for url check
            if brand_url_hint in json_text.lower():
                return brand
            if brand_at_name in json_text:
                return brand
            if brand in json_text:
                return brand

    # Nothing found, return empty string
    return ""


def build_features():
    '''
    Builds feautures needed to cluster twitter users togeher

    Currently, given a twitter json object, extracts:
       userid
       screen name
       name
       total tweets
       friends count
       followers count
       likes
       Time Inc Brands Twitted About - each being a binary feature with 1 being talked about Brand

    Stores results in data/processed/twitter_data_features.csv
    '''
    # Stores all features for each userid in the raw twitter data, format {userid:{features:values}}
    twitter_data_extracted_features = {}

    # Cycle through Raw Twitter JSON data
    for twitter_json_object in raw_twitter_data.iget_json_data_entry():
        # Find mentioned brand in tweet first because skip this tweet if cannot find a Time Inc. Brand
        # use seperate function to extract brand due to the complexity / wanting to unit test it seperately
        metioned_brand = extract_brand(twitter_json_object)
        if not metioned_brand:
            continue

        # set up memory space to store extracted features
        userid = twitter_json_object['user']['id']
        if userid in twitter_data_extracted_features:
            features_set = twitter_data_extracted_features[userid]
        else:
            features_set = {}
            twitter_data_extracted_features[userid] = features_set

            features_set["screen name"] = twitter_json_object['user']["screen_name"]
            features_set["name"] = twitter_json_object['user']["name"]

        # get user features such as total tweets, etc. Store only the best results
        features_set["total tweets count"] = max(twitter_json_object['user']["statuses_count"], features_set.get("total tweets count", 0))
        features_set["friends count"] = max(twitter_json_object['user']["friends_count"], features_set.get("friends count", 0))
        features_set["followers count"] = max(twitter_json_object['user']["followers_count"], features_set.get("followers count", 0))
        features_set["likes"] = max(twitter_json_object['user']["favourites_count"], features_set.get("likes", 0))

        # create binary brand has been metioned by user features if not already present in features_set
        if metioned_brand not in features_set:
            for brand in brands_twitter_info_table:
                features_set[brand] = 0
        features_set[metioned_brand] = 1

    # save results
    header_added = False
    brands_list = brands_twitter_info_table.keys() # create list so ordering of brands is consistent
    with open("data/processed/twitter_data_features.csv", 'w') as twitter_data_features_file:
        # create header
        twitter_data_features_file.write("userid,screen name,name,total tweets count,frends count,followers count")
        for brand in brands_list:
            twitter_data_features_file.write("," + brand)

        # add data
        for userid, features in twitter_data_extracted_features.iteritems():
            twitter_data_features_file.write('\n' + str(userid))

            twitter_data_features_file.write(',' + str(features["screen name"]))
            twitter_data_features_file.write(',' + str(features["name"].replace(',', '').encode('utf8')))
            twitter_data_features_file.write(',' + str(features["total tweets count"]))
            twitter_data_features_file.write(',' + str(features["friends count"]))
            twitter_data_features_file.write(',' + str(features["followers count"]))
            twitter_data_features_file.write(',' + str(features["likes"]))

            for brand in brands_list:
                twitter_data_features_file.write(',' + str(features[brand]))


if __name__ == '__main__':
    build_features()