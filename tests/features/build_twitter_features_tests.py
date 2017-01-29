import csv
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

    def test_build_features(self):
        '''
        Test build_features results
        '''
        build_features()

        # Test Data Entries
        with open("data/processed/twitter_data_features.csv", 'r') as twitter_data_features_file:
            csv_reader = csv.reader(twitter_data_features_file)

            # skip header line
            next(csv_reader, None)

            for data_entry in csv_reader:
                # Test Data Entry 1 - 24 parts in entry
                self.assertEqual(len(data_entry), 24)

                # Test Date Entry 2 - At least one brand feature is 1
                temp_sum = 0
                for brand_binary_feature_str in data_entry[7:]:
                    if brand_binary_feature_str.strip() == '1':
                        temp_sum += 1
                self.assertGreaterEqual(temp_sum, 1)

                # Test Data Entry 3 - Make sure all one brand features are 1 or 0
                for brand_binary_feature_str in data_entry[7:]:
                    if brand_binary_feature_str.strip() != '1' and brand_binary_feature_str.strip() != '0':
                        # Force Failure if this condition is true
                        self.assertEqual(0, 1)

                # Test Data Entry 4 - Check that likes, follower count, etc. twitter info are ints and >= 0
                for twitter_cat_count in data_entry[3:7]:
                    if int(twitter_cat_count) < 0:
                        self.assertEqual(0, 1)

if __name__ == '__main__':
    unittest.main()
