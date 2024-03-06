import unittest
from Simulation import Simulation
from FES import FES 
from Event import Event
from Queue import Queue
import numpy as np
from scipy import stats

class TestSimulation(unittest.TestCase):
    def initialization(self):
        self.arrival_rate = 2
        self.departure_rate = [3, 4]
        self.num_servers = 2
        self.theta = 0.4
        self.max_time = 10000      # To test the functionality of the while loop, let's just limit the time so that only one event is looked at
        self.simulation = Simulation(self.arrival_rate, self.departure_rate, self.num_servers, self.theta, self.max_time)
        
        return self.simulation, self.simulation.lam, self.simulation.mu, self.simulation.num_servers, self.simulation.theta, self.simulation.T, self.simulation.fes
    
    def test_initialization(self):
        _, lam, mu, num_servers, theta, T, fes = TestSimulation.initialization(self)
        
        self.assertEqual(lam, self.arrival_rate)
        self.assertEqual(mu, self.departure_rate)
        self.assertEqual(num_servers, self.num_servers)
        self.assertEqual(theta, self.theta)
        self.assertEqual(T, self.max_time)
        self.assertIsInstance(fes, FES)
    
    def test_simulation_random_dispatcher(self):        
        sim, lam, mu, num_servers, theta, T, fes = TestSimulation.initialization(self)
        
        for event in fes:
            counter = 0
            # Check whether the simulation logic works correctly 
            if (event.type == Event.ARRIVAL and sim.status_for_test[counter] == "accepted" and sim.queues_for_test[counter] == 1):
                counter += 1
                
                # If the event is an arrival and it is accepted to an idle server, the next action should create a departure event for this arrival 
                self.assertTrue(fes[counter].type == Event.DEPARTURE)
            elif ((event.type == Event.ARRIVAL and sim.status_for_test[counter] == "accepted" and sim.queues_for_test[counter] == 0) or 
                  (event.type == Event.ARRIVAL and sim.status_for_test[counter] == "rejected")):
                counter += 1
                
                # If the event is an arrival and it is accepted to a non-idle server, the next action cannot create a departure event for this arrival, but we should move forward with the next arrival
                # Alternatively, if the event is an arrival and it is rejected, there is nothing to send to the queues so the next action must be handling the next arrival as well
                self.assertTrue(fes[counter].type == Event.ARRIVAL)
            elif (event.type == Event.DEPARTURE and sim.queues_for_test[counter] > 0):
                counter += 1
                
                # If the event is a departure, and there are customers waiting for the server, we should generate another depature for the waiting customers
                self.assertTrue(fes[counter].type == Event.DEPARTURE)
             
if __name__ == '__main__':
    unittest.main()