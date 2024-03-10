import numpy as np
from typing import List, Tuple, Dict



class SARSA():
    def __init__(self, alpha, epsilon, lr, num_states: int, num_actions: int):
        self.alpha: float = alpha  # discount factor
        self.epsilon: float = epsilon  # exploration probabilities
        self.lr: float = lr  # learning rate
        self.num_states: int = num_states
        self.num_actions: int = num_actions

        self.Q = np.full((num_states, num_actions), -np.infty)  #TODO: I DONT KNOW WHAT STRUCTURE TO GIVE THIS

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.num_actions+1)
        else:
            return np.argmax(self.Q[state]) # STRUCTURE WILL HAVE IMPACT HERE

    def update(self, state: int, action: int, reward: float, next_state: int, next_action: int):
        """Update the Q-value for the given state-action pair."""
        update_value = self.Q[state, action]
        new_value = self.Q[next_state, next_action]

        # SARSA update rule
        update_value = (1 - self.lr) * (update_value) + self.lr * (reward + self.alpha * new_value)
        # new_value = (1 - self.lr) * (old_value) + self.lr *(reward + self.alpha * next_max)
        self.Q[state, action] = update_value





def main():
    sarsa = SARSA(
        alpha=0.5,
        epsilon=0.1,
        lr=0.01,
        num_states=5,
        num_actions=2
    )



