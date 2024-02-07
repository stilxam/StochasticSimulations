import numpy as np
import random


class Queue:
    def __init__(self, length, width):
        self.q = np.array((length, width))

    def __len__(self):
        return self.q.shape[0]

    def width(self):
        return self.q.shape[1]

    def head(self):
        h = self.q[1, :]
        self.q = self.q[1:, :]
        return h
