from aux_functions.Boat import Boat
from aux_functions.Queue import Queue
# from aux_functions.Boat import dynamic_group_assignment
import numpy as np


def single_line(queue: Queue, boat_capacity):
    """
    This function will simulate the process of filling a boat with boat_capacity
    :param queue: takes a queue object
    :param boat_capacity: Maximum capacity of the boat
    :return: the queue after the boat has been filled, and the number of people in the boat
    """
    boat = Boat(n=boat_capacity)
    # boat occupancy
    boat_occupancy = 0

    while boat.filled_seats <= boat_capacity:
        if len(queue) == 0:
            break
        group = queue.head()
        if boat.is_filling_possible(group):
            boat.fill_boat(group)

        elif not boat.is_filling_possible(group):
            queue.stack(group)
            break
    # TODO: RETURN THE NUMBER OF PEOPLE IN THE BOAT
    return queue, boat.filled_seats


def two_lines(queue: Queue, boat_capacity):
    """
    This function will simulate the process of filling a boat with boat_capacity with a normal queue and a queue of singles
    :param queue: Queue object
    :param boat_capacity: Maximum capacity of the boat
    :return: Queue after the boat has been filled, and the number of people in the boat
    """
    boat = Boat(n=boat_capacity)

    # boat occupancy
    boat_occupancy = 0

    while boat.filled_seats <= boat_capacity:
        if len(queue) == 0:
            break
        group = queue.head()
        # we give priority to groups over single riders
        # however if the next group can't fit in the boat, we will give priority to single riders 
        if ((not group == 1) and (boat.is_filling_possible(group))):
            boat.fill_boat(group)
        elif not boat.is_filling_possible(group):
            queue.stack(group)
            if queue.is_singles() and boat.is_filling_possible(1):
                single_rider = queue.pop_singles()
                boat.fill_boat(single_rider)
            else:
                break
    # TODO: RETURN THE NUMBER OF PEOPLE IN THE BOAT
    return queue, boat.filled_seats


def dynamic_queue(queue: Queue, boat_capacity: int):
    """
    Given queue (a list of groups), and a boat capacity, uses dynamic programing to return a subset of groups that can fit in the
    boat while maximizing the number of boat_capacity in the boat

    Args:
        queue (Queue): The queue of boat_capacity, where each element is a group of boat_capacity.
        boat_capacity (int): The maximum capacity of the boat.

    Returns:
        tuple: A tuple containing two elements:
            - A list of the most fitted boat_capacity that can fit in the boat.
            - The total number of the boat_capacity in the boat.
    """
    queue_local_copy = queue.copy()
    size_of_queue = len(queue_local_copy)

    m = np.empty((size_of_queue + 1, boat_capacity + 1))

    for i in range(size_of_queue + 1):
        for j in range(boat_capacity + 1):
            if (i == 0 or j == 0):
                m[i, j] = 0
            elif (queue_local_copy.q[i - 1] <= j):
                # update the available capacity
                updated_j = j - queue_local_copy.q[i - 1]
                m[i, j] = max(queue_local_copy.q[i - 1] + m[i - 1, int(updated_j)], m[i - 1, j])
            else:
                m[i, j] = m[i - 1, j]

    # Backtracking to find the groups that fit in the boat
    groups = []
    i = size_of_queue
    j = boat_capacity
    while i > 0 and j > 0:
        if m[i, j] != m[i - 1, j]:
            groups.append(queue_local_copy.q[i - 1])
            j -= int(queue_local_copy.q[i - 1])
        i -= 1

    # for each group that fit in the boat, remove it from the queue
    for group in groups:
        # find the index of the first occurrence of the group in the queue
        index = np.where(queue.q == group)[0][0]
        # remove the group from the queue
        queue.q = np.delete(queue.q, index)

    return queue, m[size_of_queue, boat_capacity]
