import pickle
import unittest

class TwitterClusteringModelTests(unittest.TestCase):
    def test_train_final_model(self):
        '''
        Tests train_final_model method

        Checks that a valid model file is created and userid cluster labels are saved in csv
        '''
        # Test 1 Valid Pickled Model Created
        with open("models/twitter_clustering_model.p", 'rb') as model_file:
            model = pickle.load(model_file)
        if not hasattr(model, "labels_") or not hasattr(model, "predict"):
            self.assertEqual(1, 0)

        # Test 2 Saved Clustet Labels
        with open("data/processed/twitter_user_cluster_labels.csv", 'r') as label_file:
            next(label_file)
            for data_line in label_file:
                split_data_line = data_line.split(',')
                if len(split_data_line) != 2:
                    self.assertEqual(1, 0)
                int(split_data_line[1])

if __name__ == '__main__':
    unittest.main()