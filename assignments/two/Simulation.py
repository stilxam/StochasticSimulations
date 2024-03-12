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
    def __init__(self, alpha, epsilon, lr, max_permited_q_length: int, num_of_possible_actions: int):
        self.alpha: float = alpha  # discount factor
        self.epsilon: float = epsilon  # exploration probabilities
        self.max_permited_q_length: int = max_permited_q_length  # number of customers in the queue (this includes the person being served)
        self.num_of_possible_actions: int = num_of_possible_actions  # number of servers we can dispatch to (plus 1 for rejecting the customer)

        # initialize the Q-table with dimensions (max_permited_q_length, max_permited_q_length, ..., max_permited_q_length, num_of_possible_actions)
        self.Q = np.full(
            tuple([max_permited_q_length for _ in range(num_of_possible_actions-1)] + [num_of_possible_actions]),
            -np.infty
        )
        self.lr = lr  # learning rate

    def update_epsilon(self, n):
        """
        Update the value of epsilon based on the given iteration number.

        Parameters:
        - n (int): The current iteration number.

        Returns:
        None
        """
        if n < 1000:
            self.epsilon = 1
        else:
            self.epsilon = max([0.99 * (1 - (n - 1000) / (10000)), 0.01])

    def choose_action(self, state: np.array) -> int:
        """
        Selects an action based on the given state.

        Parameters:
        - state (np.array): The current state of the simulation.

        Returns:
        - int: The chosen action.

        """
        if np.random.uniform(0, 1) < self.epsilon:
            # random action (i.e. choose random server to dispatch to)
            return np.random.choice(list(range(self.num_of_possible_actions)))
        else:
            # choose the action with the highest Q-value
            return int(np.argmax(self.Q[tuple(state)]))

    def update(self, previous_state: np.array, previous_action: int, reward: float, current_state: np.array, current_action: int):
        """
        Updates the Q-value based on the SARSA update rule.

        Args:
            previous_state (np.array): The previous state of the environment.
            previous_action (int): The previous action taken in the environment.
            reward (float): The reward received after taking the previous action.
            current_state (np.array): The current state of the environment.
            current_action (int): The current action taken in the environment.

        Returns:
            None
        """

        # convert the state and action to a tuple (to be used as an index in the Q-table)
        try:
            previous_state_tuple = tuple(np.concatenate((previous_state, previous_action), dtype=int))
        except:
            previous_state_tuple = tuple(np.concatenate((previous_state, [previous_action]), dtype=int))

        update_value = self.Q[previous_state_tuple]

        try:
            current_state_tuple = tuple(np.concatenate((current_state, current_action), dtype=int))
        except:
            current_state_tuple = tuple(np.concatenate((current_state, [current_action]), dtype=int))
        new_value = self.Q[current_state_tuple]

        # SARSA update rule
        update_value = (1 - self.lr) * update_value + self.lr * (reward + self.alpha * new_value)

        # update the Q-value for the previous state-action pair
        self.Q[previous_state_tuple] = update_value

    def get_reward(self, xis, queues, current_time, previous_time):
        """
        Calculates the reward based on the number of customers in each queue.

        Parameters:
        - xis (list): target number of customers we aim to have in each queue.
        - queues (list): A list of server objects representing the queues.
        - current_time (float): The current time.
        - previous_time (float): The previous time.

        Returns:
        - reward (float): The reward value based on the number of customers in each queue.
        """
        # array of whether the number of customers in each queue is equal to the xi of that queue
        arr = [server.number_of_customers == xis[i] for i, server in enumerate(queues)]

        # if all the queues have the number of customers equal to their xi, return the time difference
        if all(arr):
            return current_time - previous_time
        else:
            return 0
        
        # --------------------------ALTERNATIVE REWARD FUNCTION---------------------
        # total_diff = sum(abs(server.number_of_customers - xis[i]) for i, server in enumerate(queues))
        # return 1 / (0.1 + total_diff)


class Simulation:
    def __init__(self, arrival_rate: float, departure_rates: list, m, theta, Max_Time):
        self.lam = arrival_rate
        self.mus = departure_rates
        self.theta = theta # probability of accepting a customer (for the random dispatcher strategy)
        self.m = m # number of servers
        self.T = Max_Time
        self.arrDist = scipy.stats.expon(scale=1 / self.lam)
        self.queues = [Queue(mu) for mu in self.mus] # the server objects
        self.results = np.zeros(self.m) # array to store the long-term average number of customers in each queue
        self.probabilities_q = np.ones(self.m) * (1 / self.m) # probabilities of dispatching to each server (for the random dispatcher strategy)
        self.fes = FES() # future event list
        self.time = 0 # current time
        self.tOld = 0 # old time

    def simulate_random_dispatcher(self):
        """
        Simulates a random dispatcher for a multi-server system using an event-driven approach.

        This method simulates the arrival and service of customers in a (multi) server system. 
        It uses an event-driven simulation approach where events are generated based on arrival and departure times. 
        The simulation runs until a specified time limit (self.T) is reached.

        Returns:
        - results (list): A list of long-term average numbers of customers in each queue after the simulation ends.

        Notes:
        -----
        The simulation process involves the following steps:
        1. Generate the first arrival event.
        2. Loop until the simulation time reaches the specified limit (self.T).
        3. Update the simulation time based on the next event.
        4. Update the area under the curve (S) for each queue based on the time elapsed and the number of customers in the queue.
        5. Handle arrival events by either accepting or rejecting the customer and dispatching them to a server.
        6. Handle departure events by processing the next customer in the queue.
        7. Schedule the next arrival and departure events based on the current state of the system.
        8. After the simulation ends, calculate the long-term average number of customers in each queue.

        The method uses a random dispatcher strategy, where customers are randomly assigned to servers upon arrival. 

        The simulation results are stored in self.results, which is a list containing the 
        long-term average number of customers in each queue.
        """

        # generate first arrival
        self.fes.add(Event(Event.ARRIVAL, self.arrDist.rvs(), -1))

        while self.time < self.T:
            # update the old time, get the next event from the FES and update the current time
            self.tOld = self.time
            event = self.fes.next()
            self.time = event.time

            # update the area under the curve for each queue
            for i in range(self.m):
                if self.queues[i].number_of_customers > 0:
                    self.queues[i].S += (self.time - self.tOld) * (self.queues[i].number_of_customers - 1)

            if event.type == Event.ARRIVAL:
                # accept or reject the customer
                status = np.random.choice(['accepted', 'rejected'], p=[self.theta, 1 - self.theta])

                if status == "accepted":
                    # randomly select a server to dispatch to
                    server_id = np.random.choice(range(self.m), p=self.probabilities_q)
                    self.queues[server_id].arrival()

                    # check if the server was idle prior to the current arrival
                    if self.queues[server_id].status == Queue.IDLE:
                        # if yes, then schedule a departure event for the the arrival event 
                        self.fes.add(
                            Event(Event.DEPARTURE, self.time + self.queues[server_id].servDist.rvs(), server_id))

                # schedule the next arrival event
                self.fes.add(Event(Event.ARRIVAL, self.time + self.arrDist.rvs(), -1))

            else:
                self.queues[event.server_id].departure()

                # if there are still customers in the queue, schedule the next departure event
                if self.queues[event.server_id].number_of_customers > 0:
                    self.fes.add(Event(Event.DEPARTURE, self.time + self.queues[event.server_id].servDist.rvs(),
                                       event.server_id))

        # calculate the long-term average number of customers in each queue
        for i in range(self.m):
            self.results[i] = self.queues[i].S / self.time

        return self.results

    def simulate_sarsa_dispatcher(self, alpha, epsilon, xis, lr, max_queue_length):
        """
        Simulates the SARSA dispatcher algorithm.

        Parameters:
        - alpha (float): The discount factor.
        - epsilon (float): The exploration probability for SARSA.
        - xis (list): List of xi values.
        - lr (float): The learning rate.
        - max_queue_length (int): The maximum permitted queue length.

        Returns:
        - results (list): List of average queue lengths.
        - Q (dict): The Q-table learned by SARSA.
        """
    def simulate_sarsa_dispatcher(self, alpha, epsilon, xis, lr, max_queue_length):

        # generate first arrival and add it to the FES (no server ID yet, hence -1)
        self.fes.add(Event(Event.ARRIVAL, self.arrDist.rvs(), -1))

        # possible actions is the number of possible servers we can dispatch to + 1 (i.e. reject the customer)
        possible_actions = self.m + 1

        xis = [xi + 1 for xi in xis]  # we add 1 to the xi to account for the person being processed

        salsa_dancer = SARSA(
            alpha=alpha,
            epsilon=epsilon,
            lr=lr,
            max_permited_q_length = max_queue_length,
            num_of_possible_actions = possible_actions
        )

        # Pass the current state to the SARSA dispatcher to choose an action
        state = np.zeros(self.m).astype(int)
        action = [np.random.choice(possible_actions)]

        # Set the first state-action pair to 0
        init_index = tuple(np.concatenate((state, action), dtype=int))
        salsa_dancer.Q[init_index] = 0

        iteration = 0

        while self.time < self.T:
            salsa_dancer.update_epsilon(iteration)
            self.tOld = self.time
            previous_state = state
            previous_action = action

            event = self.fes.next()
            self.time = event.time


            # update the area under the curve for each queue
            for i in range(self.m):
                if self.queues[i].number_of_customers > 0:
                    self.queues[i].S += (self.time - self.tOld) * (self.queues[i].number_of_customers - 1)

            if event.type == Event.ARRIVAL:
                # choose an action based on the current system state (i.e. number of customers in each queue)
                action = salsa_dancer.choose_action([queue.number_of_customers for queue in self.queues])

                # applying action
                if action != self.m:
                    # consequences of the dispatch action
                    self.queues[action].arrival()
                    # check if the server was idle prior to the current arrival
                    if self.queues[action].status == Queue.IDLE:
                        self.fes.add(
                            Event(Event.DEPARTURE, self.time + self.queues[action].servDist.rvs(), action))

                # person is dispatched, hence, state is updated
                state = [queue.number_of_customers for queue in self.queues]

                if salsa_dancer.Q[tuple(np.concatenate((state, [action]), dtype=int))] == -np.infty:
                    salsa_dancer.Q[tuple(np.concatenate((state, [action]), dtype=int))] = 0

                # calculate the reward based on the number of customers in each queue
                reward = salsa_dancer.get_reward(xis, self.queues, self.time, self.tOld)

                # update the Q-table based on the previous state-action pair and the current state-action pair
                # we only update the Q-value before the dispatcher makes its routing decision
                salsa_dancer.update(
                    previous_state,
                    previous_action,
                    reward,
                    state,
                    action
                )

                iteration += 1

                # schedule the next arrival event
                self.fes.add(Event(Event.ARRIVAL, self.time + self.arrDist.rvs(), -1))

            else:
                self.queues[event.server_id].departure()
                # if there are still customers in the queue, schedule the next departure event
                if self.queues[event.server_id].number_of_customers > 0:
                    self.fes.add(Event(Event.DEPARTURE, self.time + self.queues[event.server_id].servDist.rvs(),
                                       event.server_id))


        for i in range(self.m):
            self.results[i] = self.queues[i].S / self.time

        return self.results, salsa_dancer.Q


    def perform_n_simulations(self, nr_runs, dispatcher: int, alpha=None, epsilon=None, lr = None, xis=None, max_queue_length=None):

        """
        Perform n simulations of the dispatcher of choice.

        Parameters:
        - nr_runs (int): The number of simulations to perform.
        - dispatcher (int): The type of dispatcher to use. 
                            0 for random dispatcher, 1 for SARSA dispatcher.
        - alpha (float): The discount factor for SARSA dispatcher.
        - epsilon (float): The exploration probability for SARSA dispatcher.
        - lr (float): The learning rate for SARSA dispatcher.
        - xis (list): List of xi values for SARSA dispatcher.
        - max_queue_length (int): The maximum permitted queue length for SARSA dispatcher.
        
        Returns:
        - sim_results (numpy.ndarray): An array containing the results of the simulations.
        """

        sim_results = []

        if dispatcher == 0:
            for _ in range(nr_runs):
                sim_results.append(self.simulate_random_dispatcher())
                       
        elif dispatcher == 1:
            for _ in range (nr_runs):
                if alpha is None or epsilon is None or lr is None or xis is None or max_queue_length is None:
                    raise ValueError("Alpha, epsilon, lr, xis, and max_queue_length must be provided for SARSA dispatcher")
                sim_results.append(self.simulate_sarsa_dispatcher(alpha, epsilon, lr, xis, max_queue_length))
        
        return np.array(sim_results)

def dancing(n_its):
    m = 2
    simulation = Simulation(arrival_rate=0.7, departure_rates=[1,1], m=m, theta=0.5, Max_Time=10000)

    results = np.empty((n_its, m))
    max_q_len = 30
    q_s = np.empty((n_its, max_q_len, max_q_len, m+1))
    for i in range(n_its):
        results[i], q_s[i] = simulation.simulate_sarsa_dispatcher(alpha=0.9, epsilon=1, lr=0.2, xis=[2,2], max_queue_length=max_q_len)
    print(f"mean: {results.mean(axis=0)}")
    print(f"std: {results.std(axis=0)}")

    print(results)


    print(f"average q_s: {q_s[0]}")


if __name__ == "__main__":
    dancing(10000)
