from aux_functions.Queue import Queue
from QueuingSystems import single_line, two_lines
import numpy as np
from joblib import Parallel, delayed
import multiprocessing as mp
import random
import time
from tqdm.auto import tqdm


def boat_line(q_Type: str, len_q, max_group_s, min_group_s, passengers):
    """
    This function will simulate the process of filling boats until the queue is empty
    TODO: add @mir implementation
    :param len_q: is the length of the queue
    :param max_group_s: is the maximum group size
    :param min_group_s: is the minimum group size
    :param passengers: Max number of passengers in the boat
    :return: the number of boats divided by the length of the queue
    """
    Q = Queue(length=len_q, high=max_group_s, low=min_group_s)
    total_its = 0
    while len(Q) > 0:
        if q_Type == "BASE":
            Q = single_line(Q, passengers)
        elif q_Type == "SINGLES":
            Q = two_lines(Q, passengers)
        else:
            raise ValueError("Invalid q_Type")

        total_its += 1
    return total_its / len_q


def stochastic_roller_coaster(
        q_Type: str = "BASE",
        n_runs: int = 100000,
        len_q: int = 100,
        max_group_s: int = 8,
        min_group_s: int = 1,
        passengers=8,
        n_jobs=mp.cpu_count() - 1
) -> np.array:
    """
    This function runs the individual simulations in parallel
    :param q_Type: takes a string defining the type of queue system
    :param n_runs: specifies the number of runs to be executed
    :param len_q: specifies the length of the queue
    :param max_group_s: specifies the maximum group size
    :param min_group_s: specifies the minimum group size
    :param passengers: specifies the number of passengers in the boat
    :param n_jobs: specifies the number of jobs to be executed in parallel
    :return: an array with the results of the simulations
    """
    t_init = time.time()

    results = Parallel(n_jobs=n_jobs)(
        delayed(boat_line)(q_Type, len_q, max_group_s, min_group_s, passengers) for _ in range(n_runs))

    t_term = time.time()
    results = np.array(results)

    print(f"\n{q_Type} TOOK  {t_term - t_init} SECONDS")
    return results
