import numpy as np
import random
import scipy
import heapq
from assignments.two.legacy_code.Event import Event

from scipy import stats
from assignments.two.legacy_code.FES import FES

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
        self.servDist = stats.expon(scale=1 / self.mu)
        
        # surface below queue length graph
        self.S = 0
        self.t = 0
        # status of the server (busy or not)
        self.status = Queue.Idle

        # stores the area of this queue after each iteration
        self.area_history = []

        # current number of customers
        self.num_customers = 0
    
    def add_area_to_history(self):
        self.area_history.append(self.S)

    def return_values(self):
        return self.num_customers, self.status, self.t, self.S


