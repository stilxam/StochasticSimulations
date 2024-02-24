import unittest
import QueueingSystems as qs
from aux_functions.Queue import Queue
import numpy as np


class TestQueueingSystems(unittest.TestCase):

    def test_single_line(self, queue: list, boat_capacity, expected_queue, 
                         expected_boat_occupancy):

        test_queue = Queue(1,1,1)
        test_queue.enqueue(queue)
        test_queue.q = np.delete(test_queue.q, 0)

        test_queue, boat_occupancy = qs.single_line(test_queue, boat_capacity)
 
        self.assertEqual(np.array(test_queue.q), np.array(expected_queue))
        self.assertEqual(boat_occupancy, expected_boat_occupancy)

    def test_two_lines(self, queue: list, boat_capacity, expected_queue, 
                       expected_boat_occupancy):

        test_queue = Queue(1,1,1)
        test_queue.enqueue(queue)
        test_queue.q = np.delete(test_queue.q, 0)

        test_queue, boat_occupancy = qs.two_lines(test_queue, boat_capacity)

        self.assertEqual(np.array(test_queue.q), np.array(expected_queue))
        self.assertEqual(boat_occupancy, expected_boat_occupancy)

    def test_dynamic_queue(self, queue: list, boat_capacity, expected_queue, 
                           expected_boat_occupancy):

        test_queue = Queue(1,1,1)
        test_queue.enqueue(queue)
        test_queue.q = np.delete(test_queue.q, 0)

        test_queue, boat_occupancy = qs.dynamic_queue(test_queue, boat_capacity)

        self.assertEqual(np.array(test_queue.q), np.array(expected_queue))
        self.assertEqual(boat_occupancy, expected_boat_occupancy)
    
    #-------------------------------------------------------------------------------#
    
    def test_single_line_1(self):
        queue = [8,7]
        boat_capacity = 8
        expected_queue = [7]
        expected_boat_occupancy = 8

        self.test_single_line(queue, boat_capacity, expected_queue, expected_boat_occupancy)

    def test_single_line_2(self):
        queue = [1,2,3,8]
        boat_capacity = 8
        expected_queue = [8]
        expected_boat_occupancy = 6

        self.test_single_line(queue, boat_capacity, expected_queue, expected_boat_occupancy)
    
    def test_single_line_3(self):
        queue = [7,8,1,1]
        boat_capacity = 8
        expected_queue = [8,1,1]
        expected_boat_occupancy = 7

        self.test_single_line(queue, boat_capacity, expected_queue, expected_boat_occupancy)
    
    # test to see that groups have priority over single riders
    def test_two_lines_1(self):
        queue = [1,1,1,1,8,7]
        boat_capacity = 8
        expected_queue = [1,1,1,1,7]
        expected_boat_occupancy = 8

        self.test_two_lines(queue, boat_capacity, expected_queue, expected_boat_occupancy)
    
    # test to see that single riders are given priority when the next group can't fit in the boat 
    def test_two_lines_2(self):
        queue = [1,1,1,6,7,1]
        boat_capacity = 8
        expected_queue = [1,7,1]
        expected_boat_occupancy = 8

        self.test_two_lines(queue, boat_capacity, expected_queue, expected_boat_occupancy)

    def test_two_lines_3(self):
        queue = [1,6,7,2]
        boat_capacity = 8
        expected_queue = [7,2]
        expected_boat_occupancy = 7

        self.test_two_lines(queue, boat_capacity, expected_queue, expected_boat_occupancy)

    def test_dynamic_queue_1(self):
        queue = [2,3,4,2]
        boat_capacity = 8
        expected_queue = [3]
        expected_boat_occupancy = 8

        self.test_dynamic_queue(queue, boat_capacity, expected_queue, expected_boat_occupancy)
    
    def test_dynamic_queue_2(self):
        queue = [2,2,1,1,2,2]
        boat_capacity = 8
        expected_queue = [2]
        expected_boat_occupancy = 8

        self.test_dynamic_queue(queue, boat_capacity, expected_queue, expected_boat_occupancy)
    
    # test to see that priority is given to groups/single riders that have in the queue the longest
    # only on the condition that this does not reduce the number of passengers in the boat
    def test_dynamic_queue_2(self):
        queue = [2,2,2,2,1,1]
        boat_capacity = 8
        expected_queue = [1,1]
        expected_boat_occupancy = 8

        self.test_dynamic_queue(queue, boat_capacity, expected_queue, expected_boat_occupancy)

if __name__ == '__main__':
    unittest.main()


