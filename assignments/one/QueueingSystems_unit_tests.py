import unittest
from QueueingSystems import single_line, two_lines, dynamic_queue
from aux_functions.Queue import Queue
import numpy as np


class TestQueueingSystems(unittest.TestCase):
    
    # def test_single_line(self, queue: list, boat_capacity, expected_queue, 
    #                      expected_boat_occupancy):

    #     test_queue = Queue(1,1,1)
    #     test_queue.enqueue(queue)
    #     test_queue = np.delete(test_queue, 0)

    #     test_queue, boat_occupancy = single_line(test_queue, boat_capacity)
    #     print("")
 
    #     self.assertEqual(np.array(test_queue), np.array(expected_queue))
    #     self.assertEqual(boat_occupancy, expected_boat_occupancy)

    # def test_two_lines(self, queue: list, boat_capacity, expected_queue, 
    #                    expected_boat_occupancy):

    #     test_queue = Queue(1,1,1)
    #     test_queue.enqueue(queue)
    #     test_queue = np.delete(test_queue, 0)

    #     test_queue, boat_occupancy = two_lines(test_queue, boat_capacity)

    #     self.assertEqual(test_queue.q, expected_queue)
    #     self.assertEqual(boat_occupancy, expected_boat_occupancy)

    # def test_dynamic_queue(self, queue: list, boat_capacity, expected_queue, 
    #                        expected_boat_occupancy):

    #     test_queue = Queue(1,1,1)
    #     test_queue.enqueue(queue)
    #     test_queue = np.delete(test_queue, 0)

    #     test_queue, boat_occupancy = dynamic_queue(test_queue, boat_capacity)

    #     self.assertEqual(test_queue.q, expected_queue)
    #     self.assertEqual(boat_occupancy, expected_boat_occupancy)

    #--------------------------------------------------------------------------#
    
    def test_single_line_1(self):
        queue = [8,7]
        boat_capacity = 8
        expected_queue = [7]
        expected_boat_occupancy = 5

        test_queue = Queue(1,1,1)
        test_queue.enqueue(queue)
        test_queue = np.delete(test_queue, 0)

        test_queue, boat_occupancy = single_line(test_queue, boat_capacity)

        self.assertEqual(np.array(test_queue), np.array(expected_queue))
        self.assertEqual(boat_occupancy, expected_boat_occupancy)

        # self.test_single_line(queue, boat_capacity, expected_queue, 
        #                       expected_boat_occupancy)
    
    # def test_two_lines_1(self):
    #     queue = [7,4,1,3]
    #     boat_capacity = 8
    #     expected_queue = [4,3]
    #     expected_boat_occupancy = 8
    #     self.test_two_lines(queue, boat_capacity, expected_queue, 
    #                           expected_boat_occupancy)

    
if __name__ == '__main__':
    unittest.main()


