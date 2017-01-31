def _iget_clicksteam_dataentry(clickstream_file):
    '''
    yields a list of clickstream userid and click features from clickstream_file

    format of data entry is [clicksteam_userid, click_feature0, ......., click_featureN]
    '''
    with open(clickstream_file, 'r') as clicksteam_data_file:
        for data_line in clicksteam_data_file:
            stripped_data_line = data_line.strip()
            if stripped_data_line:
                split_data_strs = stripped_data_line.split(",")
                yield [split_data_strs[0]] + [int(data_str) for data_str in split_data_strs[1:]]
            

def iget_clickstream_dataentry_part0():
    '''
    Returns a data entry iterator for data in data/raw/clickstream_data/UserSessions_2016_11_15_part0.csv
    
    format of data entry is [clicksteam_userid, click_feature0, ......., click_featureN]

    i.e. the part 1 out of 2 of the features 
    '''
    return _iget_clicksteam_dataentry("data/raw/clickstream_data/UserSessions_2016_11_15_part0.csv")


def iget_clickstream_dataentry_part1():
    '''
    Returns a data entry iterator for clickstream data in data/raw/clickstream/UserSessions_2016_11_15_part1.csv

    format of data entry is [clicksteam_userid, click_feature0, ......., click_featureN]

    i.e. the part 2 out of 2 of the features 
    '''
    return _iget_clicksteam_dataentry("data/raw/clickstream_data/UserSessions_2016_11_15_part1.csv")

