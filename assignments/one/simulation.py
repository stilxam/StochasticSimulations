from typing import Tuple

from aux_functions.Queue import Queue
from QueuingSystems import single_line, two_lines, dynamic_queue
import numpy as np
from joblib import Parallel, delayed
import multiprocessing as mp
import random
import time
from tqdm.auto import tqdm
from evaluation import confidence_interval, matplotlib_plot_results, queue_plot_results
import os


def simulate_boat_line(q_type: str, len_q, max_group_s, min_group_s, boat_capacity, max_time_interval: int, min_q_s,
                       max_q_s):
    """
    Simulates the queueing system for a boat ride.

    Args:
        q_type (str): The type of queueing system to simulate. Possible values are "BASE", "SINGLES", and "DYNAMIC".
        len_q (int): The length of the queue.
        max_group_s (int): The maximum group size.
        min_group_s (int): The minimum group size.
        boat_capacity   (int): The number of boat_capacity . 
        max_time_interval (int): The number of time intervals to be simulated (i.e. the number of boats to be filled)

    Returns:
        tuple: A tuple containing the mean queue length and mean boat occupancy.
            If q_type is "SINGLES", the tuple also contains the mean length of the singles queue,
            mean length of the regular queue, and mean total queue length.
    """

    # Stores the length of the queue at the end of each time interval.
    # The value at index i is the length of the queue at the end of time interval i.
    queue_length_per_interval = np.zeros(max_time_interval)

    # Stores the number of people in the boat at the end of each time interval.
    boat_occupancy_per_interval = np.zeros(max_time_interval)

    # Stores the number of people in the singles queue at the end of each time interval.
    singles_queue_length_per_interval = np.zeros(max_time_interval)

    # Stores the number of people in the regular queue at the end of each time interval.
    regular_queue_length_per_interval = np.zeros(max_time_interval)

    # Initialize queue.
    Q = Queue(length=len_q, high=max_group_s, low=min_group_s)
     

    # Time interval iterator.
    i = 0

    while max_time_interval > 0:
        if q_type == "BASE":
            Q, boat_occupancy = single_line(Q, boat_capacity)
        elif q_type == "SINGLES":
            Q, boat_occupancy = two_lines(Q, boat_capacity)
        elif q_type == "DYNAMIC":
            Q, boat_occupancy = dynamic_queue(Q, boat_capacity)
             

        if q_type in ["BASE", "DYNAMIC"]:
            # Calculate the number of people in the queue.
            for group in Q.q:
                queue_length_per_interval[i] += group
        elif q_type == "SINGLES":
            # Calculate the number of people in the singles and regular queues.
            for group in Q.q:
                if group == 1:
                    singles_queue_length_per_interval[i] += group
                else:
                    regular_queue_length_per_interval[i] += group

        # Store the occupancy of the boat at the end of the time interval.
        boat_occupancy_per_interval[i] = boat_occupancy

        # # Generate new arrivals for next iteration and add them to the end of the queue.
        # rng = np.random.default_rng(12345)
        # len_q = rng.integers(min_q_s, max_q_s)

        possible_group_arrivals = [0,1,2,3]
        probabilities = [0.1,0.2,0.3,0.4]
        len_q = np.random.choice(possible_group_arrivals, p=probabilities)

        new_arrival = Queue(length=len_q, high=max_group_s, low=min_group_s)
        Q.enqueue(new_arrival.q)
        # Update the time interval and iterator.
        max_time_interval -= 1
        i += 1

    if q_type == "SINGLES":
        # Calculate the mean queue length of the singles queue.
        mean_length_singles = np.mean(singles_queue_length_per_interval)

        # Calculate the mean queue length of the regular queue.
        mean_length_regular = np.mean(regular_queue_length_per_interval)

        # Calculate the mean boat occupancy.
        mean_boat_occupancy = np.mean(boat_occupancy_per_interval)

        # Calculate the mean total queue length.
        total_queue = singles_queue_length_per_interval + regular_queue_length_per_interval
        mean_total_queue_length = np.mean(total_queue)

        return mean_length_singles, mean_length_regular, mean_total_queue_length, mean_boat_occupancy, np.nan

    else:
        # Calculate the mean queue length.
        mean_queue_length = np.mean(np.array(queue_length_per_interval))
        # Calculate the mean boat occupancy.
        mean_boat_occupancy = np.mean(boat_occupancy_per_interval)

        return np.nan, np.nan, np.nan, mean_boat_occupancy, mean_queue_length


# def stochastic_roller_coaster(
#         q_type: str = "BASE",
#         n_runs: int = 10,
#         len_q: int = 10,
#         max_group_s: int = 8,
#         min_group_s: int = 1,
#         boat_capacity = 8,
#         max_time_interval: int = 1000,
#         n_jobs=mp.cpu_count() - 1
# ) -> np.array:
#     """
#     This function runs the individual simulations in parallel
#     :param q_type: takes a string defining the type of queue system
#     :param n_runs: specifies the number of runs to be executed
#     :param len_q: specifies the length of the queue
#     :param max_group_s: specifies the maximum group size
#     :param min_group_s: specifies the minimum group size
#     :param boat_capacity: specificies the capacity of the boat
#     :param n_jobs: specifies the number of jobs to be executed in parallel
#     :return: an array with the average number of groups per boat of the simulations
#     """
#     t_init = time.time()

#     results = Parallel(n_jobs=n_jobs)(
#         delayed(simulate_boat_line)(q_type, len_q, max_group_s, min_group_s, boat_capacity, max_time_interval)  for _ in range(n_runs))

#     t_term = time.time()
#     results = np.array(results)

#     print(f"\n{q_type} TOOK  {round(t_term - t_init, 2)} SECONDS")
#     return results

def stochastic_roller_coaster(
        n_runs: int = 10000,
        max_group_s: int = 5,
        min_group_s: int = 1,
        max_q_s: int = 5,
        min_q_s: int = 0,
        boat_capacity = 8,
        max_time_interval: int = 1000,
        n_jobs=mp.cpu_count() - 1
) -> dict:
    """
    This function runs the individual simulations in parallel for each queue type
    :param n_runs: specifies the number of runs to be executed
    :param len_q: specifies the length of the queue
    :param max_group_s: specifies the maximum group size
    :param min_group_s: specifies the minimum group size
    :param boat_capacity: specificies the capacity of the boat
    :param n_jobs: specifies the number of jobs to be executed in parallel
    :return: a dictionary where the keys are queue types and the values are arrays with the average number of groups per boat of the simulations
    """
    queue_types = ["BASE", "SINGLES", "DYNAMIC"]
    results = {}

    t_init = time.time()

    #
    # rng = np.random.default_rng(12345)
    # len_q = rng.integers(min_q_s, max_q_s)

    possible_group_arrivals = [0,1,2,3]
    probabilities = [0.1,0.2,0.3,0.4]
    len_q = np.random.choice(possible_group_arrivals, p=probabilities)


    # Create a list of tasks for all queue types
    tasks = [(q_type, len_q, max_group_s, min_group_s, boat_capacity, max_time_interval, min_q_s, max_q_s) for q_type in
             queue_types for _
             in range(n_runs)]

    # Run all tasks in parallel
    output = Parallel(n_jobs=n_jobs)(delayed(simulate_boat_line)(*task) for task in tasks)

    # Split the output by queue type

    for i, q_type in enumerate(queue_types):
        results[q_type] = np.array(output[i * n_runs:(i + 1) * n_runs])

    print("Queue Length Confidence Intervals: \n")
    print("\nBASE")
    base_q_ci: Tuple[float, float] = confidence_interval(results["BASE"][:, 4])
    print("\nSINGLES Total") 
    singles_q_t_ci: Tuple[float, float] = confidence_interval(results["SINGLES"][:, 2])
    print("\nSINGLES Regular")
    singles_q_r_ci: Tuple[float, float] = confidence_interval(results["SINGLES"][:, 1])
    print("\nSINGLES Single")
    singles_q_s_ci: Tuple[float, float] = confidence_interval(results["SINGLES"][:, 0])
    print("\nDYNAMIC")
    dynamic_q_ci: Tuple[float, float] = confidence_interval(results["DYNAMIC"][:, 4])

    queue_plot_results(
        "Queue Length",
        results["BASE"][:, 4],
        base_q_ci,
        results["SINGLES"][:, 2],
        singles_q_t_ci,
        results["SINGLES"][:, 0],
        singles_q_s_ci,
        results["SINGLES"][:, 1],
        singles_q_r_ci,
        results["DYNAMIC"][:, 4],
        dynamic_q_ci,
    ).show()



    print("Boat Filling Confidence Intervals: \n")
    print("\nBASE")
    base_boat_ci: Tuple[float, float] = confidence_interval(results["BASE"][:, 3])
    print("\nSINGLES Total")
    singles_boat_ci: Tuple[float, float] = confidence_interval(results["SINGLES"][:, 3])
    print("\nDYNAMIC")
    dynamic_boat_ci: Tuple[float, float] = confidence_interval(results["DYNAMIC"][:, 3])

    matplotlib_plot_results(
        "Seats Filled Per Boat ",
        results["BASE"][:, 3],
        base_boat_ci,
        results["SINGLES"][:, 3],
        singles_boat_ci,
        results["DYNAMIC"][:, 3],
        dynamic_boat_ci,
    ).show()

    t_term = time.time()

    print(f"\nALL SIMULATIONS TOOK  {round(t_term - t_init, 2)} SECONDS")

    # Print the results in a presentable fashion
    # for q_type, result in results.items():
    #     print(f"\nQUEUE TYPE: {q_type}")
    #     print(f"RESULTS: {result}")

    return results


# # # we just used this to see if the functions work
def main():
    max_group_s: int = 5
    min_group_s: int = 1
    boat_capacity = 8
    max_q_s: int = 5,
    min_q_s: int = 0,

    possible_group_arrivals = [0,1,2,3]
    probabilities = [0.1,0.2,0.3,0.4]
    len_q = np.random.choice(possible_group_arrivals, p=probabilities)

    # # store return values of simulate_boat_queue with q_type = "BASE"
    # m_length, m_boat_occupancy = simulate_boat_line("BASE", len_q, max_group_s, min_group_s, boat_capacity ,  1000)

    # print("Base", m_length, m_boat_occupancy)
    # # store return values of simulate_boat_queue with q_type = "SINGLES"
    # m_length_singles, m_length_regular, m_boat_occupancy, m_total_queue_length = simulate_boat_line("SINGLES", len_q, max_group_s, min_group_s, boat_capacity ,  1000)
    # print("SINGLES", m_length_singles, m_length_regular, m_boat_occupancy, m_total_queue_length )

    # store return values of simulate_boat_queue with q_type = "DYNAMIC"
    n, n, n, m_boat_occupancy, m_length = simulate_boat_line("DYNAMIC", len_q, max_group_s, min_group_s, boat_capacity ,  1000, min_q_s, max_q_s)
    print("Dynamic", m_length, m_boat_occupancy, "\n")

if __name__ == "__main__":
    if not os.path.exists('FIGURES'):
        os.makedirs('FIGURES')
    r = stochastic_roller_coaster()


