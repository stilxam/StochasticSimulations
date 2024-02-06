import numpy as np
import multiprocessing as mp
import random
import time
from joblib import Parallel, delayed


def circle_area(num_est=10000, radius=1):
    array = np.random.uniform(0, 1, (2,num_est))
    abs_dist = np.sqrt(np.sum(np.square(array), axis=0))
    is_in_circle = abs_dist <= 1
    return np.sum(is_in_circle)*4/num_est

def run_area_circle_sequentially(n_runs=100000, num_est=10000, r=1):
    results = np.zeros(n_runs)
    t1 = time.time()
    for i in range(n_runs):
        results[i] = circle_area(num_est,r)
    avg_results = results.mean()
    results_var = results.var()
    t2 = time.time()
    print(f"The sequential results are : mean = {avg_results}, var = {results_var}, it took {t2-t1} seconds")

def multithread_area(n_runs=100000, num_est=10000, r=1, num_cores = mp.cpu_count()-1):
    "TODO: FIX"
    t1 = time.time()
    results = Parallel(num_cores)(delayed(circle_area)(num_est, r) for _ in range(n_runs))
    results = np.array(results)
    avg_results = results.mean()
    results_var = results.var()
    t2 = time.time()
    print(f"The sequential results are : mean = {avg_results}, var = {results_var}, it took {t2-t1} seconds")


if __name__ == "__main__":
    run_area_circle_sequentially()
    multithread_area()




