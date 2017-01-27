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

    # Final Approach - search raw string for brand url, companye names, etc. brute force search!
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

    return ""

def build_features():
    '''
    Builds feautures needed to cluster twitter users togeher

    Currently, given a twitter json object, extracts:
       twitter userid -
       followers count -
       likes count -
       tweets count -
       time active -
       Time Inc Brands Twitted About - each being a binary feature with 1 being talked about Brand

    Stores results in data/processed/twitter_data_features.csv
    '''
    for twitter_json_object in raw_twitter_data.iget_json_data_entry():
        brand = extract_brand(twitter_json_object)


if __name__ == '__main__':
    build_features()