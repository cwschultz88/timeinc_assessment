from src.data.raw.clickstream import iget_clickstream_dataentry_part0, iget_clickstream_dataentry_part1
import unittest

class ClickstreamRawDataTests(unittest.TestCase):
    def test_iget_clickstream_dataentry_part0(self):
        '''
        Tests the iget_clickstream_dataentry_part0 method to make sure the function is actually importing clickstream data entries 

        Takes about 20 seconds to run
        '''
        count = 0
        for data_entry in iget_clickstream_dataentry_part0():
            self.assertGreaterEqual(len(data_entry), 2500)
            count += 1

        # should contain at least 1 million entries
        self.assertGreater(count, 100000)


    def test_iget_clickstream_dataentry_part1(self):
        '''
        Tests the iget_clickstream_dataentry_part1 method to make sure the function is actually importing clickstream data entries 

        Takes about 20 seconds to run
        '''
        count = 0
        for data_entry in iget_clickstream_dataentry_part1():
            self.assertGreaterEqual(len(data_entry), 2500)
            count += 1

        # should contain at least 1 million entries
        self.assertGreater(count, 100000)

if __name__ == '__main__':
    unittest.main()
