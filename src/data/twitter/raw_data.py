import json
import os


def iget_json_data_entry():
    '''
    Iterates over raw Twitter json data and yields a new json
    data object for each Twitter entry
    '''
    raw_twitter_data_directory_contents = os.listdir("data/raw/twitter_data")

    for filename in raw_twitter_data_directory_contents:
        #skip none json files as these are not data files
        if not '.json' in filename:
            continue

        # load json data objects from file
        with open(os.path.abspath("data/raw/twitter_data/" + filename)) as json_datafile:
            # each line is an individual json object
            for line in json_datafile:
                yield json.loads(line)