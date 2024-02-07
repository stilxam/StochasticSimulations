import numpy as np
import random


class Queue:
    def __init__(self, length, high, low):
        if high == low:
            self.q = high * np.ones(length)
        else:
            self.q = np.random.randint(high=high, low=low, size=length)

    def __len__(self):
        return self.q.shape[0]

    def head(self):
        h = self.q[0]
        self.q = self.q[1:]
        return h

    def __str__(self):
        return f"{self.q}"


def main():
    q = Queue(2, 5, 1)
    print(q)
    print(q.head())
    print(q)
    print(q.head())
    print(q)


if __name__ =="__main__":
    main()
