import scipy

class Queue:

    def __init__(self, departure_rate: float):
        self.mu = departure_rate
        self.servDist = scipy.stats.expon(scale=1 / self.mu)
        self.number_of_customers = 0
        self.S = 0

    def arrival(self):
        self.number_of_customers += 1
    