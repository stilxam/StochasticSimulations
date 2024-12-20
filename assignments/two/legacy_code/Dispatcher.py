import scipy 
from scipy import stats
from assignments.two.legacy_code.Event import Event
import numpy
import numpy as np

class Dispatcher:
    """
    Purpose: Decides whether to accept or reject an arrival based on theta.
    Attributes: Number of servers (num_servers), and theta.
    Methods:
    __init__: Initializes the dispatcher with given parameters.
    dispatcher: Returns the server ID and status (accepted or rejected) based on theta.
    """
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
            # server = scipy.stats.randint.rvs(0, self.num_servers)
            # server = np.random.uniform(0, self.num_servers)
            queue_probabilities = [1 / self.num_servers] * self.num_servers
            server = np.random.choice(range(self.num_servers), p = queue_probabilities)
            return server , "accepted"
        else:
            # reject the arrival
            server = -1
            return server, "rejected"

            

