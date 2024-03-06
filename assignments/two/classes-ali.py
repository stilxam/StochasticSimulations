import numpy as np
from scipy import stats
import heapq

class Queue:
    
    IDLE = 0
    BUSY = 1

    def __init__(self, mu):
        self.queue_length = 0
        self.mu = mu
        self.servDist = stats.expon(scale = 1 / self.mu)
        self.area = 0
        self.server_state = self.IDLE
    
    def handle_arrival(self):
        self.queue_length += 1
        if self.queue_length != 1:
            self.server_state = self.BUSY
    
    def handle_departure(self):
        self.queue_length -= 1
        self.server_state = self.IDLE
    
    def server_is_idle(self):
        return self.server_state == self.IDLE

class Event:
    def __init__(self, type, time, queue_id):
        self.type = type
        self.time = time
        self.queue_id = queue_id
    
    def __lt__(self, other):
        return self.time < other.time
    
    def __repr__(self):
        return f'Type: {self.type}, Time: {self.time}, Queue: {self.queue_id}'

class FES:
    def __init__(self):
        self.events = []

    def add(self, event):
        heapq.heappush(self.events, event)
    
    def next(self):
        return heapq.heappop(self.events)
    
    def is_empty(self):
        return len(self.events) == 0
    
    def __repr__(self):
        s = ''
        sorted_events = sorted(self.events)
        for event in sorted_events:
            s += f'{event} \n'
        return s
    
class Simulation:
    def __init__(self, mus, lam, theta, total_time):
        self.mus = mus
        self.lam = lam
        self.theta = theta
        self.old_time = 0
        self.current_time = 0
        self.total_time = total_time
        self.arrDist = stats.expon(scale = 1 / self.lam)
        self.queues = [Queue(mu) for mu in self.mus]
        self.results = [0] * len(self.mus)

    def simulate(self):
        future_events = FES()
        future_events.add(Event('arrival', self.arrDist.rvs(), len(self.mus) + 1))

        while self.current_time < self.total_time:
            self.old_time = self.current_time
            next_event = future_events.next()
            self.current_time = next_event.time

            for i in range(len(self.queues)):
                self.queues[i].area += 0 if self.queues[i].queue_length == 0 else (self.current_time - self.old_time) * (self.queues[i].queue_length - 1)

            if next_event.type == 'arrival':
                decision = np.random.choice(['accept', 'reject'], p = [self.theta, 1 - self.theta])
                
                if decision == 'accept':
                    queue_probabilities = [1 / len(self.mus)] * len(self.mus)
                    queue_choice = np.random.choice(range(len(self.mus)), p = queue_probabilities)
                    self.queues[queue_choice].handle_arrival()
                    if self.queues[queue_choice].server_is_idle():
                        future_events.add(Event('departure', self.current_time + self.queues[queue_choice].servDist.rvs(), queue_choice))

                future_events.add(Event('arrival', self.current_time + self.arrDist.rvs(), len(self.mus) + 1))
            else:
                self.queues[next_event.queue_id].handle_departure()
                if self.queues[next_event.queue_id].queue_length > 0:
                    future_events.add(Event('departure', self.current_time + self.queues[next_event.queue_id].servDist.rvs(), next_event.queue_id))
        
        for i in range(len(self.mus)):
            self.results[i] = self.queues[i].area / self.current_time

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
    total_time = 10000
    np.random.seed(12345)
    results_queues = [[] for _ in range(len(mus))]
    for theta in thetas:
        simulation = Simulation(mus, lam, theta, total_time)
        results = simulation.perform_multiple_runs(10000)
        print(f'----- Results for theta = {theta} -----')
        for i in range(len(mus)):
            print(f"Sample mean of long term average number of customers for queue {i + 1} = {np.mean(results, axis = 0)[i]}")
        print('')
        print(f'----- Confidence Intervals for theta = {theta} -----')
        print(simulation.calculate_CI(results))
        print('')

if __name__ == "__main__":
    main()