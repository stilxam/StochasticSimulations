from aux_functions.Queue import Queue
from QueuingSystems import single_line, two_lines, dynamic_queue
import numpy as np
from joblib import Parallel, delayed
import multiprocessing as mp
import random
import time
from tqdm.auto import tqdm

# def boat_line(q_Type: str, len_q, max_group_s, min_group_s, passengers):
#     """
#     This function will simulate the process of filling boats until the queue is empty
#     TODO: add @mir implementation
#     :param len_q: is the length of the queue
#     :param max_group_s: is the maximum group size
#     :param min_group_s: is the minimum group size
#     :param passengers: Max number of passengers in the boat
#     :return: the number of groups per boat
#     """
#     Q = Queue(length=len_q, high=max_group_s, low=min_group_s)
#     total_its = 0
#     while len(Q) > 0:
#         if q_Type == "BASE":
#             Q = single_line(Q, passengers)
#         elif q_Type == "SINGLES":
#             Q = two_lines(Q, passengers)
#         else:
#             raise ValueError("Invalid q_Type")

#         total_its += 1
#     return len_q/ total_its

def boat_line_continuous(q_Type: str, len_q, max_group_s, min_group_s, passengers, max_time_interval: int):
    '''
    This function will simulate the process of filling boats with the groups in the queue.
    Our aim is to fill each boat with the maximum number of passengers possible. 
    After filling each boat, we will take three measures per time interval:
    1. number of people in the queue at the end of the time interval (queue length)
    2. number of people in the boat at the end of the time interval
    3. number of groups in the queue at the end of the time interval
    A time one inteval the time between the departures of two consecutive boats. 
    For example: if there are 5 people waiting at the end of time slot t-1, and 
    in time slot t one group arrives of size 2 and one group arrives of size 3, 
    then the boat departing at the end of time slot t will be filled with 7 people 
    (the 5 waiting people plus the group of two persons) and the queue length at 
    the end of time slot t will be equal to 3

    :param len_q: is the length of the queue
    :param max_group_s: is the maximum group size
    :param min_group_s: is the minimum group size
    :param passengers: Max number of passengers in the boat
    :param max_time_interval: the number of time intervals to be simulated (i.e. the number of boats to be filled)

    return: number of people in the queue at the end of each time interval, 
    number of people in the boat at the end of each time interval, 
    number of groups in the queue at the end of each time interval
    
    '''

    # list that stores the lenght of the queue at the end of each time slot.
    # queue lenth is the number of people in the queue at the end of each time slot
    # value at index i is the length of the queue at the end of time slot i
    queue_length_per_interval = np.zeros(max_time_interval)

    # list that stores the number of people in the boat at the end of each time slot.
    boat_occupancy_per_interval = np.zeros(max_time_interval)

    # list that stores the number of groups in the queue at the end of each time slot.
    groups_in_queue_per_interval = np.zeros(max_time_interval)

    # stores the number of people in the singles queue at the end of each time slot
    singles_queue_length_per_interval = np.zeros(max_time_interval)

    # stores the number of people in the regular queue at the end of each time slot
    regular_queue_length_per_interval = np.zeros(max_time_interval)

    # initialize queue
    Q = Queue(length=len_q, high=max_group_s, low=min_group_s)

    # iterator
    i = 0


    while max_time_interval > 0:

        if q_Type == "BASE":
            Q, boat_occupancy = single_line(Q, passengers)

        elif q_Type == "SINGLES":
            Q, boat_occupancy = two_lines(Q, passengers)
        
        elif q_Type == "DYNAMIC":
            Q, boat_occupancy = dynamic_queue(Q, passengers)
        

        if (q_Type == "BASE" or q_Type == "DYNAMIC"):

            # calculate the number of people in the queue at the end of the time interval
            # iterate through the queue and sum the number of people in each group
            for group in Q.q:
                queue_length_per_interval[i] += group
        
        elif q_Type == "SINGLES":

            # calculate the number of people in the singles and regular queues at the end of the time interval
            for group in Q.q:
                if group == 1:
                    singles_queue_length_per_interval[i] += group
                else:
                    regular_queue_length_per_interval[i] += group

        # store the number of groups in the queue at the end of the time interval
        groups_in_queue_per_interval[i] = len(Q)
                    
        # store the occupancy of the boat at the end of the time interval
        boat_occupancy_per_interval[i] = boat_occupancy

        # generate new arrivals for next iteration and add them to the end of the queue
        rng = np.random.default_rng(12345)
        len_q = rng.integers(0,10)

        new_arrival = Queue(length=len_q, high=max_group_s, low=min_group_s)
        # qsize = new_arrival.q
        # print("")
        Q.enqueue(new_arrival.q)

        #decrease the time interval
        max_time_interval -= 1

        # increase the iterator
        i += 1

    if (q_Type == "SINGLES"):
        # calculate the mean queue length of the singles queue
        mean_length_singles = np.mean(singles_queue_length_per_interval)

        # calculate the mean queue length of the regular queue
        mean_length_regular = np.mean(regular_queue_length_per_interval)

        # calculate the mean boat occupancy
        mean_boat_occupancy = np.mean(boat_occupancy_per_interval)

        # calculate the mean total queue length
        total_queue = np.concatenate([singles_queue_length_per_interval, regular_queue_length_per_interval])
        mean_total_queue_length = np.mean(total_queue)


        return mean_length_singles, mean_length_regular, mean_boat_occupancy, mean_total_queue_length
    
    else:
        # calculate the mean que length
        mean_queue_length = np.mean(queue_length_per_interval)

        # calculate the mean boat occupancy
        mean_boat_occupancy = np.mean(boat_occupancy_per_interval)
        
        return mean_queue_length, mean_boat_occupancy

# def stochastic_roller_coaster(
#         q_Type: str = "BASE",
#         n_runs: int = 100000,
#         len_q: int = 100,
#         max_group_s: int = 8,
#         min_group_s: int = 1,
#         passengers=8,
#         n_jobs=mp.cpu_count() - 1
# ) -> np.array:
#     """
#     This function runs the individual simulations in parallel
#     :param q_Type: takes a string defining the type of queue system
#     :param n_runs: specifies the number of runs to be executed
#     :param len_q: specifies the length of the queue
#     :param max_group_s: specifies the maximum group size
#     :param min_group_s: specifies the minimum group size
#     :param passengers: specifies the number of passengers in the boat
#     :param n_jobs: specifies the number of jobs to be executed in parallel
#     :return: an array with the average number of groups per boat of the simulations
#     """
#     t_init = time.time()

#     results = Parallel(n_jobs=n_jobs)(
#         delayed(boat_line)(q_Type, len_q, max_group_s, min_group_s, passengers) for _ in range(n_runs))

#     t_term = time.time()
#     results = np.array(results)

#     print(f"\n{q_Type} TOOK  {round(t_term - t_init, 2)} SECONDS")
#     return results


def main():
    q_Type: str = "BASE"
    n_runs: int = 100000
    len_q: int = 100
    max_group_s: int = 8
    min_group_s: int = 1
    passengers=8

    rng = np.random.default_rng(12345)
    len_q = rng.integers(0,10)

    # store return values of boat_line_continuous with q_Type = "BASE"
    m_length, m_boat_occupancy = boat_line_continuous("BASE", len_q, max_group_s, min_group_s, passengers, 1000)
    
    print("Base", m_length, m_boat_occupancy)
    # store return values of boat_line_continuous with q_Type = "SINGLES"
    m_length_singles, m_length_regular, m_boat_occupancy, m_total_queue_length = boat_line_continuous("SINGLES", len_q, max_group_s, min_group_s, passengers, 1000)
    print("SINGLES", m_length_singles, m_length_regular, m_boat_occupancy, m_total_queue_length )

    # store return values of boat_line_continuous with q_Type = "DYNAMIC"
    m_length, m_boat_occupancy = boat_line_continuous("DYNAMIC", len_q, max_group_s, min_group_s, passengers, 1000)
    print("Dynamic", m_length, m_boat_occupancy)

if __name__ == "__main__":
    main()