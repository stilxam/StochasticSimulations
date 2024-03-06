from scipy import stats
import unittest
from Queue import Queue

class TestQueue(unittest.TestCase):
    def test_queue_initialization(self):
        q = Queue(2)
        self.assertEquals(q.mu, 2)
        self.assertEquals(q.status, q.Idle)
        self.assertEquals(q.S, 0)
        self.assertEquals(q.t, 0)
        self.assertEquals(q.num_customers, 0)
        
#-------------------------------------------------------------#
if __name__ == '__main__':
    unittest.main()