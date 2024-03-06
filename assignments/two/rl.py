import numpy as np
from typing import List, Tuple, Dict



class SARSA():
    def __init__(self, alpha, epsilon, lr, num_states: int, num_actions: int):
        self.alpha: float = alpha  # discount factor
        self.epsilon: float = epsilon  # exploration probabilities
        self.lr: float = lr  # learning rate

        self.num_states: int = num_states
        self.num_actions: int = num_actions

        self.Q = np.full((num_states, num_actions), -np.infty)  #state action

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.Q[state])

    def update(self, state: int, action: int, reward: float, next_state: int):
        """Update the Q-value for the given state-action pair."""
        old_value = self.Q[state, action]
        next_max = np.max(self.Q[next_state])

        # SARSA update rule
        new_value = (1 - self.lr) * (old_value) + self.lr *(reward + self.alpha * next_max)
        self.Q[state, action] = new_value

    def learn(self):
        #TODO IMPLEMENT LEARNING
        pass
