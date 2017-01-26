from src.data.raw.twitter import iget_json_data_entry
import unittest

class TwitterRawDataTests(unittest.TestCase):
    def test_iget_json_data_entry(self):
        '''
        Tests the iget_json_data_entry method to make sure the function is actually importing raw twitter data json objects

        Takes about 20 seconds to run
        '''
        twitter_json_raw_entries = [json_object for json_object in iget_json_data_entry()]

        # test that json objects were extracted from json files
        # each json file contains about 840 entries so make sure got more than just one json file
        self.assertGreater(twitter_json_raw_entries, 1000)

if __name__ == '__main__':
    unittest.main()