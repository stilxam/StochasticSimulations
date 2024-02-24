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
        expected = (1.760, 4.240)
        result = evaluation.confidence_interval(values, confidence)
        result = (round(result[0], 3), round(result[1], 3))
        self.assertEqual(result, expected)
    
    # test case 4: negative values as input
    def test_confidence_interval_4(self):
        values = np.array([-1, -2, -3, -4, -5])
        confidence = 0.05
        expected = (-4.240, -1.760)
        result = evaluation.confidence_interval(values, confidence)
        result = (round(result[0], 3), round(result[1], 3))
        self.assertEqual(result, expected)
    
    # test case 5: mix of positive and negative values as input
    def test_confidence_interval_6(self):
        values = np.array([-1, 2, -3, 4, -5])
        confidence = 0.05
        expected = (-3.459, 2.259)
        result = evaluation.confidence_interval(values, confidence)
        result = (round(result[0], 3), round(result[1], 3))
        self.assertEqual(result, expected)

    # test case 6: testing different confidence level
    def test_confidence_interval_4(self):
        values = np.array([1, 2, 3, 4, 5])
        confidence = 0.1
        expected = (1.960, 4.040)
        result = evaluation.confidence_interval(values, confidence)
        result = (round(result[0], 3), round(result[1], 3))
        self.assertEqual(result, expected)

if __name__ == "__main__":
    unittest.main()