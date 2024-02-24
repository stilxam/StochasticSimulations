import unittest
import numpy as np
import evaluation

class TestEvaluation(unittest.TestCase):
    
    # test case 1: empty array as input
    # should return (nan, nan)
    def test_confidence_interval_1(self):
        values = np.array([])
        confidence = 0.05
        expected = (np.nan, np.nan)
        result = evaluation.confidence_interval(values, confidence)
        self.assertTrue(np.isnan(result[0]) and np.isnan(result[1]) and np.isnan(expected[0]) and np.isnan(expected[1]))
    
    # test case 2: single value as input
    def test_confidence_interval_2(self):
        values = np.array([10])
        confidence = 0.05
        expected = (10, 10)
        self.assertEqual(evaluation.confidence_interval(values, confidence), expected)
    
    # test case 3: multiple values as input
    def test_confidence_interval_3(self):
        values = np.array([1, 2, 3, 4, 5])
        confidence = 0.05
        expected = (1.7604099353908769, 4.239590064609123)
        self.assertEqual(evaluation.confidence_interval(values, confidence), expected)
    
    # test case 4: negative values as input
    def test_confidence_interval_4(self):
        values = np.array([-1, -2, -3, -4, -5])
        confidence = 0.05
        expected = (-4.239590064609123, -1.7604099353908769)
        self.assertEqual(evaluation.confidence_interval(values, confidence), expected)
    
    # test case 5: mix of positive and negative values as input
    def test_confidence_interval_6(self):
        values = np.array([-1, 2, -3, 4, -5])
        confidence = 0.05
        expected = (-3.459130002367346, 2.2591300023673457)
        self.assertEqual(evaluation.confidence_interval(values, confidence), expected)

    # test case 6: testing different confidence level
    def test_confidence_interval_4(self):
        values = np.array([1, 2, 3, 4, 5])
        confidence = 0.1
        expected = (1.9597032242488852, 4.040296775751115)
        self.assertEqual(evaluation.confidence_interval(values, confidence), expected)

if __name__ == "__main__":
    unittest.main()