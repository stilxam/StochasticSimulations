from aux_functions.Boat import Boat
from aux_functions.Queue import Queue

def single_line(queue: Queue, passengers):
    """
    This function will simulate the process of filling a boat with passengers
    :param queue: takes a queue object
    :param passengers: Max number of passengers in the boat
    :return: the queue after the boat has been filled
    """
    boat = Boat(n=passengers)
    n_its = 0
    while n_its <= passengers:
        if len(queue) == 0:
            break
        group = queue.head()
        if boat.is_filling_possible(group):
            boat.fill_boat(group)
            n_its += 1
        elif not boat.is_filling_possible(group):
            queue.stack(group)
            break
    return queue
def two_lines(queue: Queue, passengers):
    """
    This function will simulate the process of filling a boat with passengers with a normal queue and a queue of singles
    :param queue: Queue object
    :param passengers: Max number of passengers in the boat
    :return: Queue after the boat has been filled
    """
    boat = Boat(n=passengers)
    n_its = 0
    while n_its <= passengers:
        if len(queue) == 0:
            break
        group = queue.head()
        if boat.is_filling_possible(group):
            boat.fill_boat(group)
            n_its += 1
        elif not boat.is_filling_possible(group):
            queue.stack(group)
            if queue.is_singles() and boat.is_filling_possible(1):
                queue.pop_singles()
                boat.fill_boat(1)
                n_its += 1
            else:
                break
    return queue