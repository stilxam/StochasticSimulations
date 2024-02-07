from aux_functions.Boat import Boat
from aux_functions.Queue import Queue
import numpy as np
from joblib import Parallel, delayed
import multiprocessing as mp
import random
import time
from tqdm.auto import tqdm

def simulate_filling_a_boat(queue: Queue, passengers):
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


def sim_queue(len_q, max_group_s, min_group_s, passengers):

    single_queue = Queue(length=len_q, high=max_group_s, low=min_group_s)
    total_its = 0
    while len(single_queue) > 0:
        single_queue = simulate_filling_a_boat(single_queue, passengers)
        total_its += 1
    return total_its/len_q


def parallelized_base_q(n_runs=100000, len_q=100, max_group_s=8, min_group_s=1, passengers=8, n_jobs=mp.cpu_count()-1):

    t_init = time.time()

    results = Parallel(n_jobs=n_jobs)(delayed(sim_queue)(len_q, max_group_s, min_group_s, passengers) for _ in range(n_runs))

    t_term = time.time()
    results = np.array(results)

    print(f"Average Iterations for Q of len {len_q} is {results.mean()}, with VAR {results.var()}")
    print(f"Process took {t_term-t_init}")
    return results



if __name__ == "__main__":
    parallelized_base_q()
