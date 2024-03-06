import numpy as np
import scipy 
import heapq

class Event:
    ARRIVAL = 0 # constant for arrival type
    DEPARTURE = -1 # constant for departure type

    def __init__(self, typ, time, id):
        self.type = typ
        self.time = time
        self.server_id = id
    
    def __lt__(self, other):
        return self.time < other.time
    
    def __repr__(self):
        s = ("Arrival", "Departure")
        return f"{s[self.type]} at {self.time} at server {self.server_id}"
    

class FES:
    
    def __init__(self):
        self.events = []
        
    def add(self, event):
        heapq.heappush(self.events, event)
        
    def next(self):
        return heapq.heappop(self.events)
    
    def isEmpty(self):
        return len(self.events) == 0
        
    def __repr__(self):
        # Note that if you print self.events, it would not appear to be sorted
        # (although they are sorted internally).
        # For this reason we use the function 'sorted'
        s = ''
        sortedEvents = sorted(self.events)
        for e in sortedEvents :
            s += f'{e}\n'
        return s

class Queue:

    IDLE = 0
    BUSY = 1

    def __init__(self, departure_rate: float):
        self.mu = departure_rate
        self.servDist = scipy.stats.expon(scale = 1 / self.mu)
        self.number_of_customers = 0
        self.S = 0
        self.status = self.IDLE
    
    def arrival(self):
        self.number_of_customers +=1
        # update the status of the server
        if self.number_of_customers > 1:
            self.status = self.BUSY
    
    def departure(self):
        self.number_of_customers -= 1
        self.status = self.IDLE

class Simulation:
    def __init__(self, arrival_rate: float, departure_rates: list, m, theta, Max_Time):
        self.lam = arrival_rate
        self.mus = departure_rates
        self.theta = theta
        self.m = m
        self.T = Max_Time
        self.arrDist = scipy.stats.expon(scale = 1 / self.lam)
        self.queues = [Queue(mu) for mu in self.mus]
        self.results = np.zeros(self.m)
        self.probabilities_q = np.ones(self.m) * (1 / self.m )
        self.fes = FES()
        self.time = 0
        self.tOld = 0
    
    def simulate(self):
        self.fes.add(Event(Event.ARRIVAL, self.arrDist.rvs(), -1))

        while self.time < self.T:
            self.tOld = self.time
            event = self.fes.next()
            self.time = event.time

            for i in range(self.m):
                if self.queues[i].number_of_customers > 0:
                    self.queues[i].S += (self.time - self.tOld) * (self.queues[i].number_of_customers - 1)
            
            if event.type == Event.ARRIVAL:
                status = np.random.choice(['accepted', 'rejected'], p = [self.theta, 1 - self.theta])
            
                if status == "accepted":
                    server_id = np.random.choice(range(self.m), p = self.probabilities_q)
                    self.queues[server_id].arrival()
                    # check if the server was idle prior to the current arrival
                    if self.queues[server_id].status == Queue.IDLE:
                        self.fes.add(Event(Event.DEPARTURE, self.time + self.queues[server_id].servDist.rvs(), server_id))
                
                self.fes.add(Event(Event.ARRIVAL, self.time + self.arrDist.rvs(), -1))
            
            else: 
                self.queues[event.server_id].departure()
                if self.queues[event.server_id].number_of_customers > 0:
                    self.fes.add(Event(Event.DEPARTURE, self.time + self.queues[event.server_id].servDist.rvs(), event.server_id))
            
        for i in range (self.m):
            self.results[i] = self.queues[i].S / self.time
        
        return self.results

                    
    def perform_multiple_runs(self, nr_runs):
        sim_results = []
        for _ in range(nr_runs):
            sim_results.append(self.simulate())
        return np.array(sim_results)

    def calculate_CI(self, sim_results):
        sample_means = np.mean(sim_results, axis = 0)
        
        sample_stds = np.std(sim_results, axis = 0, ddof = 1)
        
        return  sample_means - (1.96 * (sample_stds / np.sqrt(len(sim_results)))), sample_means + (1.96 * (sample_stds / np.sqrt(len(sim_results))))

def main():
    mus = [4, 5, 6, 7, 8]
    lam = 3
    thetas = [0.60, 0.85]
    m = 5
    total_time = 1000
    np.random.seed(12345)
    results_queues = [[] for _ in range(len(mus))]
    for theta in thetas:
        simulation = Simulation(lam, mus, m, theta, total_time)
        results = simulation.perform_multiple_runs(1000)
        print(f'----- Results for theta = {theta} -----')
        for i in range(len(mus)):
            print(f"Sample mean of long term average number of customers for queue {i + 1} = {np.mean(results, axis = 0)[i]}")
        print('')
        print(f'----- Confidence Intervals for theta = {theta} -----')
        print(simulation.calculate_CI(results))
        print('')

if __name__ == "__main__":
    main()