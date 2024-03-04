import numpy as np
from scipy import stats
import time
import threading
from Event import Event
from Queue import Queue
from Dispatcher import Dispatcher


class Simulation:
    def __init__(self, arrival_rate: int, departure_rate: list, num_servers: int, theta: int, max_time: int):
        self.lam = arrival_rate
        self.mu = departure_rate
        self.num_servers = num_servers
        self.arrDist = stats.expon(scale = 1 / self.lam)
        self.theta = theta
        self.T = max_time
        self.S = np.zeros(self.num_servers)
    
    def simulate(self):
        t = 0   # current time

        # create queues
        queues = []
        for i in range (self.num_servers):
            queues.append(Queue(self.mu[i]))

        # initialize the dispatcher
        dispatcher = Dispatcher(self.theta, self.num_servers)

        # generate the first arrival
        a = self.arrDist.rvs()
        firstEvent = Event(Event.ARRIVAL, a)

        # accept or reject the arrival
        server, status = dispatcher.dispatcher()
        if status == "accepted":
           # add the event to the queue[server]
            queues[server].add_event(firstEvent)
            # serve the event
            # queues[server].simulate()
            queues[server].serve()
        
        t += a

        while t < self.T:
            a = self.arrDist.rvs()
            arr = Event(Event.ARRIVAL, t + a)
            # accpet or reject the arrival
            server, status = dispatcher.dispatcher()
            if status == "accepted":
            # add the event to the queue[status]
                queues[server].add_event(arr)
                # serve the event
            # queues[server].simulate()
            queues[server].serve()

            t += a
    
        # calculate the area for all queues
        for i in range (self.num_servers):
            self.S[i] = queues[i].return_area()
            # self.S[i] = queues[i].average_area()

        print (self.S)

if __name__ == "__main__":
    np.random.seed(123)
    sim = Simulation (2, [3,4], 2, 0.4, 100)
    sim.simulate()
