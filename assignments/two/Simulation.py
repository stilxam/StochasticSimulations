import numpy as np
from scipy import stats
import time
import threading
from legacy_code.Event import Event
from legacy_code.Queue import Queue
from legacy_code.Dispatcher import Dispatcher
import heapq
from legacy_code.FES import FES
from rl import SARSA


class Simulation:
    def __init__(self, arrival_rate: float, departure_rate: list, num_servers: int, theta: float, max_time: int):
        self.lam = arrival_rate
        self.mu = departure_rate
        self.num_servers = num_servers
        self.arrDist = stats.expon(scale=1 / self.lam)
        self.theta = theta
        self.T = max_time
        self.fes = FES()

    def simulate(self):
        t = 0  # current time

        # create queues
        queues = []
        for i in range(self.num_servers):
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
            for i in range(self.num_servers):
                if (queues[i].num_customers == 0):
                    queues[i].S += (t - tOld) * queues[i].num_customers
                    queues[i].add_area_to_history()

                elif (queues[i].num_customers > 0):
                    queues[i].S += (t - tOld) * (queues[i].num_customers - 1)
                    queues[i].add_area_to_history()

            if event.type == Event.ARRIVAL:

                # accept or reject the arrival
                server_id, status = dispatcher.dispatcher()

                if status == "accepted":
                    queues[server_id].num_customers += 1

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

                # check if there are customes waiting for the server
                if queues[server_id].num_customers > 0:
                    dep = Event(Event.DEPARTURE, t + queues[server_id].servDist.rvs(), server_id)
                    self.fes.add(dep)

        results = []
        area_histories = []
        # print the surface below the queue length graph for all queues
        for i in range(self.num_servers):
            results.append(queues[i].S / t)
            area_histories.append(queues[i].area_history)

        return results  # area_histories

    def get_state(self, queues):
        return np.array([cust.num_customers for cust in queues])

    def apply_action_and_get_reward(self, xi, action, queue):
        arrival_time = self.arrDist.rvs()
        arrival_event = Event(Event.ARRIVAL, self.T + arrival_time, -1)
        self.fes.add(arrival_event)

        if action == 1:
            queue.num_customers += 1
        else:
            pass

        # Calculate the reward based on the outcome of the action
        # im not sure about this reward function
        reward = abs(xi - queue.num_customers)
        return queue, reward

    def sarsa_loop(self, alpha, epsilon, gamma, lr, num_states, num_actions):
        t = 0

        queues = []
        for i in range(self.num_servers):
            queues.append(Queue(self.mu[i]))

        dispatcher = SARSA(alpha, epsilon, lr, num_states, num_actions)
        # generate the first arrival
        a = self.arrDist.rvs()
        # we set the server_id to -1 to indicate that the event is not assigned to any server
        firstEvent = Event(Event.ARRIVAL, a, -1)

        # add the first event to the FES
        self.fes.add(firstEvent)

        # Pass the current state to the dispatcher to choose an action
        state = self.get_state(queues)
        action = np.random.choice(num_actions + 1)

        # Set the first state-action pair to 0
        dispatcher.Q[state, action] = 0

        while t < self.T:
            tOld = t
            event = self.fes.next()
            t = event.time

            # update the surface below the queue length graph for all queues
            for i in range(self.num_servers):
                if (queues[i].num_customers == 0):
                    queues[i].S += (t - tOld) * queues[i].num_customers
                    queues[i].add_area_to_history()

                elif (queues[i].num_customers > 0):
                    queues[i].S += (t - tOld) * (queues[i].num_customers - 1)
                    queues[i].add_area_to_history()

            # Apply the action and observe the next state and reward
            queues, reward = self.apply_action_and_get_reward(gamma, action, queues)

            # Pass this new state to the dispatcher to choose an action
            next_state = self.get_state(queues)
            next_action = dispatcher.choose_action(next_state)

            if dispatcher.Q[next_state, next_action] == -np.infty:
                dispatcher.Q[next_state, next_action] = 0

            # Update the Q-value based on the observed reward
            dispatcher.update(state, action, reward, next_state, next_action)

            # Generate the next event
            a = self.arrDist.rvs()
            arr = Event(Event.ARRIVAL, t + a, -1)
            self.fes.add(arr)

    def single_sarsa(self, alpha, epsilon, xi, lr, num_states, num_actions):
        """
        :param alpha: discount factor
        :param epsilon: exploration probability
        :param xi: goal
        :param lr: learning rate
        :param num_states: number of states
        :param num_actions: number of actions
        :return:
        """

        t = 0
        queue = Queue(self.mu[0])

        salsa_dancer = SARSA(
            alpha=alpha,
            epsilon=epsilon,
            lr=lr,
            num_states=num_states,
            num_actions=num_actions
        )

        a = self.arrDist.rvs()
        # we set the server_id to -1 to indicate that the event is not assigned to any server
        firstEvent = Event(Event.ARRIVAL, a, -1)

        # add the first event to the FES
        self.fes.add(firstEvent)

        # Pass the current state to the dispatcher to choose an action
        state = queue.num_customers
        action = np.random.choice(num_actions + 1)

        # Set the first state-action pair to 0
        salsa_dancer.Q[state, action] = 0

        while t < self.T:
            start_t = t
            previous_state = state
            previous_action = action

            event = self.fes.next()
            t = event.time()

            if queue.num_customers == 0:
                queue.S += (t - start_t) * queue.num_customers
                queue.add_area_to_history()

            elif queue.num_customers > 0:
                queue.S += (t - start_t) * (queue.num_customers - 1)
                queue.add_area_to_history()

            action = salsa_dancer.choose_action(queue.num_customers)

            queue, reward = self.apply_action_and_get_reward(xi=xi, action=action, queue=queue)

            state = queue.num_customers

            salsa_dancer.update(
                previous_state,
                previous_action,
                reward,
                state,
                action
            )


if __name__ == "__main__":
    np.random.seed(None)
    sim = Simulation(2, [3, 4], 2, 0.4, 10000)
    sim.simulate()
