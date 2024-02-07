import numpy as np
import random


class Queue:
    def __init__(self, length, high, low):
        self.q = np.random.randint(high=high, low=low, size=length)

    def __len__(self):
        return self.q.shape

    def head(self):
        h = self.q[1]
        self.q = self.q[1:]
        return h
