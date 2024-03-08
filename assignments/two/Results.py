from Simulation import Simulation
import numpy as np
from typing import Tuple
from scipy import stats
import multiprocessing as mp
from joblib import delayed, Parallel
from functools import partial
import time
import math


def confidence_interval(results):
    """
    Calculates 0.95 CI
    :param results:
    :return:
    """
    results = np.array(results)
    mean = results.mean(axis=0)
    std_dev = results.std(axis=0, ddof=1)

    low = mean - (1.96 * (std_dev / np.sqrt(len(results))))
    high = mean + (1.96 * (std_dev / np.sqrt(len(results))))

    return low, high


def main():
    """
    Purpose: Executes the simulation for different theta values and prints the results along with their confidence intervals.
    Steps:
    Initializes random seed and simulation parameters.
    Loops over different theta values, simulating the queueing system for each and collecting results.
    Calculates the mean of results and the confidence interval.
    Prints the results for each queue.
    Measures and prints the total execution time
    :return:
    """
    np.random.seed(42069)
    confidence_val = 0.05
    lambda_param = 3
    mus = [4, 5, 6, 7, 8]
    thetas = [0.60, 0.85]
    m = 5
    n_jobs = mp.cpu_count() - 1
    its = 100

    t1 = time.time()

    # results = []

    for theta in thetas:

        sim = Simulation(lambda_param, mus, m, theta, 1000)
        results = Parallel(n_jobs=n_jobs)(delayed(sim.simulate)() for _ in range(its))

        results = np.array(results)
        mean = results.mean(axis=0)

        # calculate CI
        low_bound, high_bound = confidence_interval(results=results)

        print(f"--------------Simulation results (theta = {theta})--------------")
        for i in range(m):
            print(
                f"Queue {i + 1} has Sample mean of the long-term average number of customers: {mean[i]} with CI: [{low_bound[i]}, {high_bound[i]}]")

    t2 = time.time()
    print(f"time: {t2 - t1}")


if __name__ == "__main__":
    main()
