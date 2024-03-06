import numpy as np
from scipy import stats
import time
import threading
from Event import Event
from Queue import Queue
from Dispatcher import Dispatcher
import heapq
from FES import FES


class Simulation:
    def __init__(self, arrival_rate: float, departure_rate: list, num_servers: int, theta: float, max_time: int):
        self.lam = arrival_rate
        self.mu = departure_rate
        self.num_servers = num_servers
        self.arrDist = stats.expon(scale = 1 / self.lam)
        self.theta = theta
        self.T = max_time
        self.fes = FES()
        self.status_for_test = []
        self.queues_for_test = []
    
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
        # we set the server_id to -1 to indicate that the event is not assigned to any server
        firstEvent = Event(Event.ARRIVAL, a, -1)
        
        # add the first event to the FES
        self.fes.add(firstEvent)

        while t < self.T:
            tOld = t
            event = self.fes.next()
            t = event.time

            # update the surface below the queue length graph for all queues
            for i in range (self.num_servers):
                if (queues[i].num_customers == 0 ):
                    queues[i].S += (t - tOld) * queues[i].num_customers
                    queues[i].add_area_to_history()

                elif (queues[i].num_customers > 0):
                    queues[i].S += (t - tOld) * (queues[i].num_customers - 1)
                    queues[i].add_area_to_history()

            
            if event.type == Event.ARRIVAL:

                # accept or reject the arrival
                server_id, status = dispatcher.dispatcher()
                
                # Data tracking for test purposes
                self.status_for_test.append(status)

                if status == "accepted":
                    queues[server_id].num_customers += 1
                    
                    # Data tracking for test purposes
                    self.queues_for_test.append(queues[server_id.num_customers])

                    # check if the server is idle
                    if queues[server_id].num_customers == 1:
                        dep = Event(Event.DEPARTURE, t + queues[server_id].servDist.rvs(), server_id)
                        self.fes.add(dep)
                
                a = self.arrDist.rvs()
                arr = Event(Event.ARRIVAL, t + a, -1)
                self.fes.add(arr)
            
            elif event.type == Event.DEPARTURE:
                # retrieve the server_id from the event
                server_id = event.server_id
                queues[server_id].num_customers -= 1
                
                self.queues_for_test.append(queues[server_id.num_customers])
                
                # check if there are customes waiting for the server
                if queues[server_id].num_customers > 0:
                    dep = Event(Event.DEPARTURE, t + queues[server_id].servDist.rvs(), server_id)
                    self.fes.add(dep)
            
        results = []
        area_histories = []
        # print the surface below the queue length graph for all queues
        for i in range (self.num_servers):
            results.append(queues[i].S / t)
            area_histories.append(queues[i].area_history)

        
        return results # area_histories

if __name__ == "__main__":
    np.random.seed(None)
    sim = Simulation (2, [3,4], 2, 0.4, 10000)
    sim.simulate()