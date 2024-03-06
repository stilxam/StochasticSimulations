from scipy import stats
import unittest
from Event import Event

class TestDispatcher(unittest.TestCase):
    def test_event_initialization(self):
        lam = 2
        self.arrDist = stats.expon(scale = 1 / lam)
        a = self.arrDist.rvs()
        e = Event(Event.ARRIVAL, a, -1)
        self.assertEquals(e.type, e.ARRIVAL)
        self.assertEquals(e.server_id, -1)
            
    def test_event_lt(self):
        lam = 2
        self.arrDist = stats.expon(scale = 1 / lam)
        a = self.arrDist.rvs()
        b = self.arrDist.rvs()
        
        e1 = Event(Event.ARRIVAL, a, -1)
        e2 = Event(Event.ARRIVAL, b, -1)
        expected = a < b
        actual = e1 < e2
        self.assertEquals(actual, expected)

#-------------------------------------------------------------#
if __name__ == '__main__':
    unittest.main()