import unittest
from Boat import Boat

class TestBoat(unittest.TestCase):
    
        def test_boat_initialization(self):
            b = Boat(8)
            self.assertEqual(b.n_seats, 8)
            self.assertEqual(b.filled_seats, 0)
    
        def test_fill_boat(self):
            b = Boat(8)
            b.fill_boat(3)
            self.assertEqual(b.filled_seats, 3)
    
        def test_is_filling_possible(self):
            b = Boat(8)
            self.assertTrue(b.is_filling_possible(8))
            self.assertTrue(b.is_filling_possible(7))
            self.assertFalse(b.is_filling_possible(9))
            self.assertTrue(b.is_filling_possible(0))
    
        def test_is_boat_full(self):
            b = Boat(8)
            self.assertFalse(b.is_boat_full())
            b.fill_boat(8)
            self.assertTrue(b.is_boat_full())

#-------------------------------------------------------------#
if __name__ == '__main__':
    unittest.main()