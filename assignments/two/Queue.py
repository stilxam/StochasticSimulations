import numpy as np
import random
import scipy
import heapq
from Event import Event

from scipy import stats
from FES import FES


class Queue:

    Busy = 1
    Idle = 0

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
        # status of the server (busy or not)
        self.status = Queue.Idle

        # current number of customers
        self.num_customers = 0

    
    def add_event(self, event):
        heapq.heappush(self.q, event)
        # if len(self.q) == 1: # this is the only event in the queue, so we generate the departure event
        #     self.setup_departure()
    
    def setup_departure(self):
        event = self.q[0]
        if event.type == Event.ARRIVAL:
            d = self.servDist.rvs()
            dep = Event(Event.DEPARTURE, event.time + d)
            heapq.heappush(self.q, dep)
    
    def get_first_event(self):
        return self.q[0]

    def length(self):
        return len(self.q)
    
    def get_status(self):
        return self.status
    
    def current_time(self):
        return self.t

    def serve(self):

        # while self.q:
        #     tOld = self.t
        #     event = heapq.heappop(self.q)
        #     self.t = event.time

        #     # update the area
        #     self.S += len(self.q) * (self.t - tOld)

        #     if event.type == Event.DEPARTURE and len(self.q) > 0:
        #         self.setup_departure()

        tOld = self.t
        event = heapq.heappop(self.q)
        self.t = event.time
        self.S += (self.t - self.tOld) * self.num_customers

        if event.type == Event.ARRIVAL:
            self.num_customers += 1
            if self.num_customers == 1:
                d = self.servDist.rvs()
                dep = Event(Event.DEPARTURE, event.time + d)
                heapq.heappush(self.q, dep)
        
        elif event.type == Event.DEPARTURE:
            self.num_customers -= 1
            if self.num_customers > 0:
                b = self.servDist.rvs()
                dep = Event(Event.DEPARTURE, event.time + d)
                heapq.heappush(self.q, dep)



            

    def return_area (self):
        return self.S / self.t
