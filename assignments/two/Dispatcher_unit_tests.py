import unittest
from Dispatcher import Dispatcher

class TestDispatcher(unittest.TestCase):
        
        def test_dispatcher_initialization(self):
            d = Dispatcher(0.4, 5)
            self.assertEqual(d.theta, 0.4)
            self.assertEqual(d.num_servers, 5)
        
        def test_dispatcher(self):
            d = Dispatcher(0.5, 5)
            server_id, status = d.dispatcher()
            self.assertTrue(-1 <= server_id <= 5)
            self.assertTrue(status == "accepted" or "rejected")

#-------------------------------------------------------------#
if __name__ == '__main__':
    unittest.main()