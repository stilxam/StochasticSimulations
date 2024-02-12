from aux_functions.Queue import Queue
import numpy as np

def most_fitted_passengers(queue: Queue, boat_capacity):
    """
    Given queue (a list of groups), and a boat capacity, uses dynamic programing to return a subset of groups that can fit in the
    boat while maximizing the number of passengers in the boat

    Args:
        queue (Queue): The queue of passengers, where each element is a group of passengers.
        boat_capacity (int): The maximum capacity of the boat.

    Returns:
        tuple: A tuple containing two elements:
            - A list of the most fitted passengers that can fit in the boat.
            - The total number of the passengers in the boat.
    """
    queue_local_copy = queue.copy()
    size_of_queue = len(queue_local_copy)

    m = np.empty((size_of_queue+1, boat_capacity+1))

    for i in range(size_of_queue+1):
        for j in range(boat_capacity+1):
            if (i == 0 or j == 0):
                m[i,j] = 0
            elif (queue_local_copy.q[i-1] <= j):
                # update the available capacity
                updated_j = j - queue_local_copy.q[i-1]
                m[i,j] = max(queue_local_copy.q[i-1] + m[i-1, int(updated_j)], m[i-1, j])
            else:
                m[i,j] = m[i-1, j]

    # Backtracking to find the groups that fit in the boat
    groups = []
    i = size_of_queue
    j = boat_capacity 
    while i > 0 and j > 0:
        if m[i, j] != m[i-1, j]:
            groups.append(queue_local_copy.q[i-1])
            j -= int(queue_local_copy.q[i-1])
        i -= 1

    # for each group that fit in the boat, remove it from the queue
    for group in groups:
        # find the index of the first occurrence of the group in the queue
        index = np.where(queue.q == group)[0][0]
        # remove the group from the queue
        queue.q = np.delete(queue.q, index)

    
    #TODO: return the updated queue with the groups that fit in the boat removed
    return queue, m[size_of_queue, boat_capacity]




#TODO: turn this into unit tests
# test_queue = Queue(length=4,high=2, low=2)

# for group in range(len(test_queue)):
#     group = test_queue.head()
# # i.e input groups
# test_groups = [2,2,2,2]

# for i in range(len(test_groups)):
#     test_queue.stack(test_groups[i])

# result = most_fitted_passengers(test_queue,6)

# print(result[0].__str__())
