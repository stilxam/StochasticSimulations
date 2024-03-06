import scipy 
from scipy import stats
from Event import Event
import numpy

class Dispatcher:

    def __init__(self, theta, num_servers: int):
        self.num_servers = num_servers
        self.theta = theta
    
    def dispatcher(self, ):
        # accept arrival with probability theta, otherwise reject
        # use numpy.ranodom.choice to simulate a Bernoulli distribution
        choice = numpy.random.choice(["accept", "reject"], p=[self.theta, 1-self.theta])

        # if scipy.stats.bernoulli.rvs(self.theta):
        if choice == "accept":
            # accept the arrival by randomly selecting one of the m servers
            server = scipy.stats.randint.rvs(0, self.num_servers)
            return server , "accepted"
        else:
            # reject the arrival
            server = -1
            return server, "rejected"

            
