import unittest
import numpy as np
from Queue import Queue

class TestQueue(unittest.TestCase):

    def test_queue_initialization(self):
        q = Queue(10, 5, 1)
        self.assertEqual(len(q), 10)
        self.assertTrue(np.all(q.q >= 1) and np.all(q.q <= 5))

    def test_len(self):
        q = Queue(5, 3, 1)
        self.assertEqual(len(q), 5)

    def test_head(self):

        q = Queue(1,1,1)
        np.delete(q.q, 0)

        test_queue = np.array([1, 2, 3, 4])
        q.q = test_queue
        head = q.head()

        # check head
        self.assertEqual(head, 1)
        # check that the length of the queue 
        self.assertEqual(len(q), 3)

    def test_is_singles(self):
        q = Queue(5, 2, 1)
        self.assertTrue(q.is_singles())
        q = Queue(5, 5, 5)
        self.assertFalse(q.is_singles())

    def test_pop_singles(self):
        queue = Queue(1,1,1)

        test_queue = np.array([2, 1, 3, 4])
        queue.enqueue(test_queue)
        queue.q = np.delete(queue.q, 0)

        queue.pop_singles()
        # q.is_singles() should return False becauase the queue no longer has any singles
        self.assertFalse(queue.is_singles())

        test_queue_2 = np.array([1, 1, 3, 4])
        queue.enqueue(test_queue_2)

        queue.pop_singles()
        # q.is_singles() should return True because we only remove one single rider, and there is still one left
        self.assertTrue(queue.is_singles())

    def test_stack(self):
        q = Queue(5, 3, 1)
        q.stack(4)
        self.assertEqual(q.q[0], 4)
        self.assertEqual(len(q), 6)

    def test_enqueue(self):
        q = Queue(5, 3, 1)
        q.enqueue(np.array([4, 5]))
        self.assertEqual(len(q), 7)
        self.assertTrue(np.array_equal(q.q[-2:], np.array([4, 5])))

if __name__ == '__main__':
    unittest.main()