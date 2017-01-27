from src.data.raw.twitter import iget_json_data_entry
from src.features.build_twitter_features import brands_twitter_info_table, build_features, extract_brand
import unittest

class BuildTwitterFeaturesTests(unittest.TestCase):
    def test_extract_brand(self):
        '''
        Tests extract_brand function
        '''

        # Test 1
        # test that brands can be extracted from 95% of the tweets data
        extracted_brands = []
        number_of_tweets = 0
        for twitter_json_object in iget_json_data_entry():
            number_of_tweets += 1
            brand = extract_brand(twitter_json_object)
            if brand:
                extracted_brands.append(brand)
        percent_of_brands_found = len(extracted_brands) / float(number_of_tweets)
        self.assertGreaterEqual(brands_twitter_info_table, 0.95)


        # Test 2
        # test to make sure brand twitter info caching is working
        self.assertGreater(brands_twitter_info_table, 0)



if __name__ == '__main__':
    unittest.main()
