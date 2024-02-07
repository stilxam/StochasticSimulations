from aux_functions.Boat import Boat
from aux_functions.Queue import Queue
import numpy as np
from joblib import Parallel, delayed
import multiprocessing as mp
import random
import time
from functools import partial
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
            break

    return queue


def sim_queue(len_q, max_group_s, min_group_s, passengers):

    single_queue = Queue(length=len_q, high=max_group_s, low=min_group_s)
    total_its = 0
    while len(single_queue) > 0:
        single_queue = simulate_filling_a_boat(single_queue, passengers)
        total_its += 1

    return total_its/len_q


def parallelized_q(n_runs=100000, len_q=100, max_group_s=8, min_group_s=8, passengers=16, n_jobs=mp.cpu_count()-1):

    t_init = time.time()
    # pre_filled = partial(sim_queue, len_q, max_group_s, min_group_s, passengers)

    # results = np.zeros(n_runs)
    # for i in tqdm(range(n_runs)):
    #     results[i] = sim_queue(len_q=len_q,max_group_s=max_group_s, min_group_s=min_group_s, passengers=passengers)

    results = Parallel(n_jobs=n_jobs)(delayed(sim_queue)(len_q, max_group_s, min_group_s, passengers) for _ in range(n_runs))
    # results = Parallel(n_jobs=nr_cores_to_use)(delayed(simulate_game)(game_sim) for _ in range(nruns))

    t_term = time.time()
    results = np.array(results)

    print(f"Average Iterations for Q of len {len_q} is {results.mean()}, with VAR {results.var()}")
    print(f"Process took {t_term-t_init}")



if __name__ == "__main__":
    # print(simulate_filling_a_boat(Queue(10, 8, 1), 8))
    parallelized_q()
