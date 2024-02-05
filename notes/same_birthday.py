from numpy import array, arange, zeros
import random
import time
from joblib import Parallel, delayed
import multiprocessing as mp

def choose_n_birthday(n):
    return random.choices(range(0, 365), k=n)


def single_experiment(n: int = 25) -> int:
    days = zeros(365)
    birthdays = choose_n_birthday(n)
    for day in birthdays:
        days[day] += 1
    return sum(days > 1)>0


def same_birthday(n: int = 25, num_trials: int = 10000) -> array:
    t_init = time.time()
    results = zeros(num_trials)
    for i in range(num_trials):
        results[i] = single_experiment(n)

    t_term = time.time()
    print(f"TIME FOR SINGLE CORE: {t_term - t_init}")
    return results

def main(n: int = 25, num_trials: int = 10000) -> None:
    results = same_birthday(n, num_trials)
    print(f"Average number of shared birthdays: {results.mean()}")


def parallelized(n: int = 25, num_trials: int = 10000, nr_cores_to_use=mp.cpu_count() - 1) -> array:
    t_init = time.time()
    results = Parallel(n_jobs=nr_cores_to_use)(delayed(single_experiment)(n) for _ in range(num_trials))
    t_term = time.time()
    print(f"TIME FOR PARALLELIZED: {t_term - t_init}")
    return array(results)

if __name__ == "__main__":
    num_trials = 1000000
    n = 25
    main(num_trials=num_trials, n=n)
    results = parallelized(n, num_trials)
    print(f"Average number of shared birthdays: {results.mean()}")

