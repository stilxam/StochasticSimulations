from scipy import stats
import unittest
from Event import Event
from FES import FES

class TestFES(unittest.TestCase):
    def test_FES(self):
        lam = 2
        self.arrDist = stats.expon(scale = 1 / lam)
        a = self.arrDist.rvs()
        e = Event(Event.ARRIVAL, a, -1)
        
        fes = FES()
        self.assertTrue(fes.isEmpty())
        
        fes.add(e)
        self.assertFalse(fes.isEmpty())
        self.assertEquals(fes.next(), e)       
        
#-------------------------------------------------------------#
if __name__ == '__main__':
    unittest.main()