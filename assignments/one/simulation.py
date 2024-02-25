from typing import Tuple

from aux_functions.Queue import Queue
from QueueingSystems import single_line, two_lines, dynamic_queue
import numpy as np
from joblib import Parallel, delayed
import multiprocessing as mp
import random
import time
from tqdm.auto import tqdm
from evaluation import confidence_interval, matplotlib_plot_results, queue_plot_results
import os


def simulate_boat_line(q_type: str, len_q, max_group_s, min_group_s, boat_capacity, max_time_interval: int, min_queue_size,
                       max_queue_size):
    """
    Simulates the boat line based on the given parameters.

    Args:
        q_type (str): The type of queue system to simulate. Possible values are "BASE", "SINGLES", and "DYNAMIC".
        len_q: The length of the initial queue.
        max_group_size: The maximum group size.
        min_group_size: The minimum group size.
        boat_capacity: The capacity of the boat.
        max_time_interval (int): The maximum number of time intervals to simulate.
        min_queue_size: The minimum number of groups arriving at each time interval.
        max_queue_size: The maximum number of groups arriving at each time interval.

    Returns:
        tuple: A tuple containing the following values:
            - mean_length_singles (float): The mean queue length of the singles queue (only for q_type = "SINGLES").
            - mean_length_regular (float): The mean queue length of the regular queue (only for q_type = "SINGLES").
            - mean_total_queue_length (float): The mean total queue length (only for q_type = "SINGLES").
            - mean_boat_occupancy (float): The mean boat occupancy.
            - mean_queue_length (float): The mean queue length (only for q_type = "BASE" and "DYNAMIC").
            - final_total_group_sizess (float): The sum of all group sizes combined.
            - final_total_groups_arrived (float): The sum of total number of groups that arrived.
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

    # Store the sum of all group sizes at the end of the simulation.
    final_total_group_sizes_sum = 0

    # Store the sum of total number of groups that arrived at the end of the simulation.
    final_total_number_of_groups_arrived = 0

    # Initialize queue.
    Q = Queue(length=len_q, high=max_group_s, low=min_group_s)

    final_total_group_sizes_sum += sum(Q.q)
    final_total_number_of_groups_arrived += len(Q.q)

    # Time interval iterator.
    i = 0
    # Number of boats to be simulated
    number_of_boats = max_time_interval

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

        # possible_group_arrivals = [0,1,2,3]
        # probabilities = [0.1,0.2,0.3,0.4] # simulation 1 - benchmark
        # probabilities = [0.2,0.6,0.1,0.1] #  simulation 2  - low number of groups per time interval
        # probabilities = [0.1,0.1,0.2,0.6] #  simulation 3  - high number of groups per time interval

        possible_group_arrivals = [1,2]
        probabilities = [0.5,0.5] #  simulation 4  - higher probabilities of smaller groups arrrving per time interval
        # probabilities = [0.5,0.5] #  simulation 5  - higher probabilities of smaller groups arrrving per time interval
        len_q = np.random.choice(possible_group_arrivals, p=probabilities)

        new_arrival = Queue(length=len_q, high=max_group_s, low=min_group_s)

        if (i+1) < number_of_boats:
            final_total_group_sizes_sum += sum(new_arrival.q)
            final_total_number_of_groups_arrived += len(new_arrival.q)

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

        return (mean_length_singles, mean_length_regular, mean_total_queue_length, mean_boat_occupancy, np.nan,
                final_total_group_sizes_sum, final_total_number_of_groups_arrived)

    else:
        # Calculate the mean queue length.
        mean_queue_length = np.mean(np.array(queue_length_per_interval))
        # Calculate the mean boat occupancy.
        mean_boat_occupancy = np.mean(boat_occupancy_per_interval)

        return (np.nan, np.nan, np.nan, mean_boat_occupancy, mean_queue_length,
                final_total_group_sizes_sum, final_total_number_of_groups_arrived)

def stochastic_roller_coaster(
        n_runs: int = 100,
        max_group_s: int = 5,
        min_group_s: int = 1,
        max_queue_size: int = 3,
        min_queue_size: int = 1,
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

    possible_group_arrivals = [0,1,2,3]
    probabilities = [0.1,0.2,0.3,0.4] # simulation 1 - benchmark
    # probabilities = [0.2,0.6,0.1,0.1] #  simulation 2  - low number of groups per time interval
    # probabilities = [0.1,0.1,0.2,0.6] #  simulation 3  - high number of groups per time interval

    # possible_group_arrivals = [1,2]
    # probabilities = [0.5,0.5] #  simulation 4  - higher probabilities of smaller groups arrrving per time interval
    # probabilities = [0.5,0.5] #  simulation 5  - higher probabilities of smaller groups arrrving per time interval

    len_q = np.random.choice(possible_group_arrivals, p=probabilities)


    # Create a list of tasks for all queue types
    tasks = [(q_type, len_q, max_group_s, min_group_s, boat_capacity, max_time_interval, min_queue_size, max_queue_size) for q_type in
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


    # only for sim 1
    # # calculate the mean size of the groups arriving at each time interval for all queue types combined
    # final_arrivals_total_group_size = np.sum(
    #     np.concatenate([
    #         results["BASE"][:, 5], 
    #         results["SINGLES"][:, 5], 
    #         results["DYNAMIC"][:, 5]
    #     ])
    # ) / (n_runs * len(queue_types) * max_time_interval)

    # # calculate the mean number of groups arriving at each time interval for all queue types combined
    # final_arrivals_mean_group_arrivals = np.sum(
    #     np.concatenate([
    #         results["BASE"][:, 6], 
    #         results["SINGLES"][:, 6], 
    #         results["DYNAMIC"][:, 6]
    #     ])
    # ) / (n_runs * len(queue_types) * max_time_interval)

    # print(f"\nMean number of groups arriving per time slot: {final_arrivals_mean_group_arrivals}")
    # print(f"Mean group size: {final_arrivals_total_group_size}")

    print(f"\nALL SIMULATIONS TOOK  {round(t_term - t_init, 2)} SECONDS")

    return results

if __name__ == "__main__":
    if not os.path.exists('FIGURES'):
        os.makedirs('FIGURES')
    r = stochastic_roller_coaster()