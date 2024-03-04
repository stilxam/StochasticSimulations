import numpy as np
import random
import scipy
import heapq
from Event import Event

from scipy import stats


class Queue:
    def __init__(self, departure_rate: int):
        self.q = []
        heapq.heapify(self.q)

        # departure rate mu
        self.mu = departure_rate
        # Service distribution
        self.servDist = stats.expon(scale = 1 / self.mu)
        
        # surface below queue length graph
        self.S = 0
        self.t = 0
    
    def add_event(self, event):
        heapq.heappush(self.q, event)
        if len(self.q) == 1: # this is the only event in the queue, so we generate the departure event
            self.setup_departure()
    
    def setup_departure(self):
        event = self.q[0]
        if event.type == Event.ARRIVAL:
            d = self.servDist.rvs()
            dep = Event(Event.DEPARTURE, event.time + d)
            heapq.heappush(self.q, dep)
    
    def serve(self):

        while self.q:
            tOld = self.t
            event = heapq.heappop(self.q)
            self.t = event.time

            # update the area
            self.S += len(self.q) * (self.t - tOld)

            if event.type == Event.DEPARTURE and len(self.q) > 0:
                self.setup_departure()
            

    def return_area (self):
        return self.S / self.t
