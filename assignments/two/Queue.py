import numpy as np
import random
import scipy
import heapq
from Event import Event

from scipy import stats
from FES import FES

import numpy


class Queue:

    Busy = 1
    Idle = 0

    def __init__(self, departure_rate: int):
        self.q = []
        heapq.heapify(self.q)

        # departure rate mu
        self.mu = departure_rate
        # Service distribution
        # self.servDist = numpy.random.exponential(scale = 1 / self.mu)
        self.servDist = stats.expon(scale = 1 / self.mu)
        
        # surface below queue length graph
        self.S = 0
        self.t = 0
        # status of the server (busy or not)
        self.status = Queue.Idle

        # current number of customers
        self.num_customers = 0

    def return_values(self):
        return self.num_customers, self.status, self.t, self.S


