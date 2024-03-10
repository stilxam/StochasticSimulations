from types import prepare_class
import numpy as np
import scipy
import heapq
from tqdm.auto import tqdm

class Event:
    ARRIVAL = 0  # constant for arrival type
    DEPARTURE = -1  # constant for departure type

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
        for e in sortedEvents:
            s += f'{e}\n'
        return s


class Queue:
    IDLE = 0
    BUSY = 1

    def __init__(self, departure_rate: float):
        self.mu = departure_rate
        self.servDist = scipy.stats.expon(scale=1 / self.mu)
        self.number_of_customers = 0
        self.S = 0
        self.status = self.IDLE

    def arrival(self):
        self.number_of_customers += 1
        # update the status of the server
        if self.number_of_customers > 1:
            self.status = self.BUSY

    def departure(self):
        self.number_of_customers -= 1
        self.status = self.IDLE





class SARSA():
    def __init__(self, alpha, epsilon, lr, num_states: int, num_actions: int):
        self.alpha: float = alpha  # discount factor
        self.epsilon: float = epsilon  # exploration probabilities
        self.num_states: int = num_states
        self.num_actions: int = num_actions

        self.Q = np.full((num_states, num_actions), -np.infty)  # TODO: I DONT KNOW WHAT STRUCTURE TO GIVE THIS
        self.lr = lr  # learning rate

    def update_epsilon(self, n):
        if n < 1000:
            self.epsilon = 1
        else:
            self.epsilon = max([0.99*(1-(n-1000)/(10000)), 0.01])
    def choose_action(self, state)-> int:
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            return int(np.argmax(self.Q[state]))

    def update(self, state: int, action: int, reward: float, next_state: int, next_action: int):
        """Update the Q-value for the given state-action pair."""
        update_value = self.Q[state, action]
        new_value = self.Q[next_state, next_action]

        # SARSA update rule
        update_value = (1 - self.lr) * (update_value) + self.lr * (reward + self.alpha * new_value)
        # new_value = (1 - self.lr) * (old_value) + self.lr *(reward + self.alpha * next_max)
        self.Q[state, action] = update_value


    def get_reward(self, xi, queues):
        return 1/(abs(xi - sum([cust.number_of_customers for cust in queues]))+0.1)


class Simulation:
    def __init__(self, arrival_rate: float, departure_rates: list, m, theta, Max_Time):
        self.lam = arrival_rate
        self.mus = departure_rates
        self.theta = theta
        self.m = m
        self.T = Max_Time
        self.arrDist = scipy.stats.expon(scale=1 / self.lam)
        self.queues = [Queue(mu) for mu in self.mus]
        self.results = np.zeros(self.m)
        self.probabilities_q = np.ones(self.m) * (1 / self.m)
        self.fes = FES()
        self.time = 0
        self.tOld = 0

    def simulate(self):
        self.fes.add(Event(Event.ARRIVAL, self.arrDist.rvs(), -1))

        i_lr = 0

        while self.time < self.T:
            self.tOld = self.time
            event = self.fes.next()
            self.time = event.time

            for i in range(self.m):
                if self.queues[i].number_of_customers > 0:
                    self.queues[i].S += (self.time - self.tOld) * (self.queues[i].number_of_customers - 1)

            if event.type == Event.ARRIVAL:
                status = np.random.choice(['accepted', 'rejected'], p=[self.theta, 1 - self.theta])

                if status == "accepted":
                    server_id = np.random.choice(range(self.m), p=self.probabilities_q)
                    self.queues[server_id].arrival()
                    # check if the server was idle prior to the current arrival
                    if self.queues[server_id].status == Queue.IDLE:
                        self.fes.add(
                            Event(Event.DEPARTURE, self.time + self.queues[server_id].servDist.rvs(), server_id))

                self.fes.add(Event(Event.ARRIVAL, self.time + self.arrDist.rvs(), -1))

            else:
                self.queues[event.server_id].departure()
                if self.queues[event.server_id].number_of_customers > 0:
                    self.fes.add(Event(Event.DEPARTURE, self.time + self.queues[event.server_id].servDist.rvs(),
                                       event.server_id))

        for i in range(self.m):
            self.results[i] = self.queues[i].S / self.time

        return self.results

    def running_rl(self, alpha, epsilon, xi, lr, num_states, num_actions):
        self.fes.add(Event(Event.ARRIVAL, self.arrDist.rvs(), -1))

        salsa_dancer = SARSA(
            alpha=alpha,
            epsilon=epsilon,
            lr = 1,
            num_states=num_states,
            num_actions=num_actions
        )
        # Pass the current state to the dispatcher to choose an action
        state = 0
        action = np.random.choice(num_actions )

        # Set the first state-action pair to 0
        salsa_dancer.Q[state, action] = 0

        iteration = 0

        while self.time < self.T:
            salsa_dancer.update_epsilon(iteration)
            self.tOld = self.time
            previous_state = state
            previous_action = action

            event = self.fes.next()
            self.time = event.time

            for i in range(self.m):
                # area under
                if self.queues[i].number_of_customers > 0:
                    self.queues[i].S += (self.time - self.tOld) * (self.queues[i].number_of_customers - 1)

            if event.type == Event.ARRIVAL:

                # TODO: dispatch happens here
                
                
                action = salsa_dancer.choose_action(self.queues[0].number_of_customers)

                # applying action
                if action == 1:
                    # consequences of the dispatch
                    # server_id = np.random.choice(range(self.m), p=self.probabilities_q)
                    self.queues[0].arrival()
                    # check if the server was idle prior to the current arrival
                    if self.queues[0].status == Queue.IDLE:
                        self.fes.add(
                            Event(Event.DEPARTURE, self.time + self.queues[0].servDist.rvs(), 0))


                # person is dispatched, hence, state is updated
                state = self.queues[0].number_of_customers
                
                if salsa_dancer.Q[state, action] == -np.infty:
                   salsa_dancer.Q[state, action] = 0
                # Reward

                reward = salsa_dancer.get_reward(xi, self.queues)
                
                salsa_dancer.update(
                    previous_state,
                    previous_action,
                    reward,
                    state,
                    action                
                )
                iteration += 1

                self.fes.add(Event(Event.ARRIVAL, self.time + self.arrDist.rvs(), -1))

            else:
                self.queues[event.server_id].departure()
                if self.queues[event.server_id].number_of_customers > 0:
                    self.fes.add(Event(Event.DEPARTURE, self.time + self.queues[event.server_id].servDist.rvs(),
                                       event.server_id))

        for i in range(self.m):
            self.results[i] = self.queues[i].S / self.time

        return self.results

    def perform_multiple_runs(self, nr_runs):
        sim_results = []
        for _ in range(nr_runs):
            sim_results.append(self.simulate())
        return np.array(sim_results)


def dancing(n_its):
    simulation = Simulation(arrival_rate=0.7, departure_rates=[1], m=1, theta=0.5, Max_Time=10000)

    results = np.empty(n_its)
    for i in tqdm(range(n_its)):
        results[i] = simulation.running_rl(alpha=0.9, epsilon=1,lr=0.2, xi=3, num_states=100, num_actions=2)[0]
    print(f"mean: {results.mean()}")
    print(f"std: {results.std()}")

if __name__ == "__main__":
    dancing(1000)