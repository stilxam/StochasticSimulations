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
        self.num_states: int = num_states  # number of customers in the queue
        self.num_actions: int = num_actions  # server that we can dispatch to

        self.Q = np.full(
            tuple([num_states for _ in range(num_actions-1)] + [num_actions]),
            -np.infty
        )
        self.lr = lr  # learning rate

    def update_epsilon(self, n):
        if n < 1000:
            self.epsilon = 1
        else:
            self.epsilon = max([0.99 * (1 - (n - 1000) / (10000)), 0.01])

    def choose_action(self, state: np.array) -> int:
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(list(range(self.num_actions)))
        else:
            return int(np.argmax(self.Q[tuple(state)]))

    def update(self, state: np.array, action: int, reward: float, next_state: np.array, next_action: int):
        """Update the Q-value for the given state-action pair."""
        try:
            tup_state = tuple(np.concatenate((state, action), dtype=int))
        except:
            tup_state = tuple(np.concatenate((state, [action]), dtype=int))

        update_value = self.Q[tup_state]


        try:
            new_tup_state = tuple(np.concatenate((next_state, next_action), dtype=int))
        except:
            new_tup_state = tuple(np.concatenate((next_state, [next_action]), dtype=int))
        new_value = self.Q[new_tup_state]

        # SARSA update rule
        update_value = (1 - self.lr) * update_value + self.lr * (reward + self.alpha * new_value)

        # new_value = (1 - self.lr) * (old_value) + self.lr *(reward + self.alpha * next_max)
        q = self.Q
        self.Q[tup_state] = update_value

    def get_reward(self, xis, queues, current_time, previous_time):
        # arr = [server.number_of_customers == xis[i] for i, server in enumerate(queues)]
        # if all(arr):
        #     return current_time - previous_time
        # else:
        #     return 0
        total_diff = sum(abs(server.number_of_customers - xis[i]) for i, server in enumerate(queues))
        return 1 / (0.1 + total_diff)


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

    def running_rl(self, alpha, epsilon, xis, lr, max_queue_length):
        self.fes.add(Event(Event.ARRIVAL, self.arrDist.rvs(), -1))
        num_actions = len(self.queues)+1

        xis = [xi + 1 for xi in xis]  # we add 1 to the xi to account for the person being processed

        salsa_dancer = SARSA(
            alpha=alpha,
            epsilon=epsilon,
            lr=lr,
            num_states=max_queue_length,
            num_actions=num_actions
        )
        # Pass the current state to the dispatcher to choose an action
        state = np.zeros(num_actions-1).astype(int)
        action = [np.random.choice(num_actions)]

        # Set the first state-action pair to 0
        init_index = tuple(np.concatenate((state, action), dtype=int))
        salsa_dancer.Q[init_index] = 0

        iteration = 0

        count = 0
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

            reward = salsa_dancer.get_reward(xis, self.queues, self.time, self.tOld)
            if reward!= 0:
                count+=1


            if event.type == Event.ARRIVAL:

                action = salsa_dancer.choose_action([queue.number_of_customers for queue in self.queues])

                # applying action
                if action != num_actions-1:
                    # consequences of the dispatch
                    # server_id = np.random.choice(range(self.m), p=self.probabilities_q)
                    self.queues[action].arrival()
                    # check if the server was idle prior to the current arrival
                    if self.queues[action].status == Queue.IDLE:
                        self.fes.add(
                            Event(Event.DEPARTURE, self.time + self.queues[action].servDist.rvs(), action))

                # person is dispatched, hence, state is updated
                state = [queue.number_of_customers for queue in self.queues]

                observation = salsa_dancer.Q[tuple(np.concatenate((state, [action]), dtype=int))]


                if salsa_dancer.Q[tuple(np.concatenate((state, [action]), dtype=int))] == -np.infty:
                    salsa_dancer.Q[tuple(np.concatenate((state, [action]), dtype=int))] = 0
                # Reward
                # reward = salsa_dancer.get_reward(xis, self.queues, self.time, self.tOld)

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
        return self.results, salsa_dancer.Q

    def perform_multiple_runs(self, nr_runs):
        sim_results = []
        for _ in range(nr_runs):
            sim_results.append(self.simulate())
        return np.array(sim_results)


def dancing(n_its):
    m = 2
    simulation = Simulation(arrival_rate=0.7, departure_rates=[1, 1], m=m, theta=0.5, Max_Time=5000)

    results = np.empty((n_its, m))
    q_s = np.empty((n_its, 8, 8, 3))
    for i in range(n_its):
        results[i], q_s[i] = simulation.running_rl(alpha=0.9, epsilon=1, lr=0.2, xis=[2,2], max_queue_length=8)
    # print(f"mean: {results.mean(axis=0)}")
    # print(f"std: {results.std(axis=0)}")


    print(f"average q_s: {q_s.mean(axis=0)}")


if __name__ == "__main__":
    dancing(100)
