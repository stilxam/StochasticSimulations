import numpy as np
import random


class Queue:
    def __init__(self, length, high, low):
        """
        Initialize a Queue object.

        Parameters:
        - length (int): The length of the queue.
        - high (int): The upper bound (inclusive) for groups in the queue.
        - low (int): The lower bound (inclusive) for groups in the queue.
        """
        self.high = high
        self.low = low
        if high == low:
            self.q = high * np.ones(length)
        else:
            self.q = np.random.randint(high=high+1, low=low, size=length)

    def __len__(self):
        """
        Get the length of the queue.

        Returns:
        - int: The length of the queue.
        """
        return self.q.shape[0]

    def head(self):
        """
        Get the head of the queue and remove it.

        Returns:
        - int: The head of the queue.
        """
        h = self.q[0]
        self.q = self.q[1:]
        return h

    def is_singles(self):
        """
        Check if there are any groups of size 1 in the queue.

        Returns:
        - bool: True if there are groups with size 1, False otherwise.
        """
        indexes = np.where(self.q == 1)[0]
        if len(indexes) > 0:
            return True
        else:
            return False

    def pop_singles(self):
        """
        Remove the first group of size 1 from the queue.

        Returns:
        - int: The value 1.
        """
        indexes = np.where(self.q == 1)[0]
        first_one = indexes[0]
        self.q = np.concatenate([self.q[:first_one], self.q[first_one + 1:]])

        return 1

    def stack(self, head):
        """
        Add a group to the front of the queue.

        Parameters:
        - head (int): The element to be added to the front of the queue.
        """
        self.q = np.concatenate([np.array([head]), self.q])

    def enqueue(self, head):
        """
        Add groups(s) to the end of the queue.

        Parameters:
        - head (int): The group(s) to be added to the end of the queue.
        """
        self.q = np.concatenate([self.q, head])

    def __str__(self):
        """
        Get a string representation of the queue.

        Returns:
        - str: A string representation of the queue.
        """
        return f"{self.q}"
    
    def copy(self):
        """
        Create a copy of the queue.

        Returns:
        - Queue: A new Queue object with the same properties and values as the original queue.
        """
        new_queue = Queue(len(self), self.high, self.low)
        new_queue.q = self.q.copy()
        return new_queue


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
