import numpy as np
import random


class Queue:
    def __init__(self, length, high, low):
        if high == low:
            self.q = high * np.ones(length)
        else:
            self.q = np.random.randint(high=high+1, low=low, size=length)

    def __len__(self):
        return self.q.shape[0]

    def head(self):
        h = self.q[0]
        self.q = self.q[1:]
        return h

    def is_singles(self):
        indexes = np.where(self.q == 1)[0]
        if len(indexes) > 0:
            return True
        else:
            return False

    def pop_singles(self):
        indexes = np.where(self.q == 1)[0]
        first_one = indexes[0]
        self.q = np.concatenate([self.q[:first_one], self.q[first_one + 1:]])
        return 1

    def stack(self, head):
        self.q = np.concatenate([np.array([head]), self.q])

    def __str__(self):
        return f"{self.q}"


def main():
    q = Queue(20, 3, 1)
    print(q)
    print(q.singles())
    print(q)
    # print(q)
    # print(q.head())
    # print(q)


if __name__ == "__main__":
    main()
