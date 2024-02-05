import multiprocessing

from numpy import mean, std, var, zeros
import random
import time
from joblib import Parallel, delayed
import multiprocessing as mp
from tqdm.auto import tqdm
import numpy as np



def simulate_game(n):
    """
    This function simulates a game of dice
    :param n: cutoff number of points
    :return (nr_Points, nr_Throws): number of points scored, number of attempts it took
    """
    nr_Throws = 0
    nr_Points = 0
    while nr_Points < n:
        nr_Throws += 1
        nr_Points += random.randint(1, 6)
    return nr_Points, nr_Throws


def main(nruns=10000, game_sim=50):
    """
    This function runs the simulation in a single core
    :param nruns:
    :param game_sim:
    :return:
    """

    results_list = zeros((nruns, 2))

    t_init = time.time()
    for i in range(nruns):
        results_list[i] = simulate_game(game_sim)
    t_term = time.time()
    print("SINGLE CORE:")
    print(f"Simulation took {t_term - t_init}")
    print(f"Simulated mean points {mean(results_list[:,0])}")
    print(f"Simulated $\sigma^2$ points {var(results_list[:,0])}")
    print(f"Simulated $\sigma$ points {std(results_list[:,0])}")


def parallelized(nruns=10000, game_sim=50, nr_cores_to_use=mp.cpu_count() - 1):
    """
    This function runs the simulation in parallel
    :param nruns:
    :param game_sim:
    :param nr_cores_to_use:
    :return:
    """
    t_init = time.time()
    results = Parallel(n_jobs=nr_cores_to_use)(delayed(simulate_game)(game_sim) for _ in range(nruns))

    t_term = time.time()
    results_np = np.array(results)
    print(f"PARALLELIZED")
    print(f"Simulation took {t_term - t_init}")
    print(f"Simulated mean points {mean(results_np[:,0])}")
    print(f"Simulated $\sigma^2$ points {var(results_np[:,0])}")
    print(f"Simulated $\sigma$ points {std(results_np[:,0])}")
    print(f"Number of cores used {nr_cores_to_use}")

if __name__ == "__main__":
    main(nruns=1000000)
    parallelized(nruns=1000000)
