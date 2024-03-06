from Simulation import Simulation
import numpy as np
from typing import Tuple
from scipy import stats
import multiprocessing as mp
from joblib import delayed, Parallel
from functools import partial
import time

def confidence_interval(results: np.array, confidence: float = 0.05) -> Tuple[float, float]:
    """
    This function will calculate the confidence interval of the results.
    :param results: the results of the simulations
    :param confidence: the confidence level
    :return: None
    """
    mean: float = results.mean()
    var: float = results.var()
    n: int = len(results)
    z: float = stats.norm.ppf(1 - (confidence) / 2)
    half_width: float = z * (var / n) ** 0.5
    
    lower: float = mean - half_width
    upper: float = mean + half_width

    print(f"MEAN: {mean} \nVAR: {var}\nSTD {var ** 0.5}\nN: {n}\nZ: {z}\nHalf Width: {half_width}")
    print(f"CI: [{lower}, {upper}]")
    return lower, upper

def main():
    np.random.seed(42069)
    confidence_val = 0.05
    lambda_param = 3
    mus = [4,5,6,7,8]
    thetas = [0.60,0.85]
    m = 5
    n_jobs = mp.cpu_count()-1
    its = 100

    t1 = time.time()

    # results = []
# 

    # sim = Simulation (lambda_param, mus, m, thetas[0], 1000)
    # results = Parallel(n_jobs=n_jobs)(delayed(sim.simulate)() for _ in range(its))



    
    sim = Simulation (lambda_param, mus, m, thetas[0], 1000)
    results = np.empty((its,m))
    for i in range (its):
        results[i] = (sim.simulate())

    t2 = time.time()    


    print(f"time: {t2-t1}")

    # for theta_param in thetas:
    #     sim = Simulation (lambda_param, mus, m, theta_param, 10000)
    #     results = sim.simulate()
    
    # print (results)
    
    # for i in range(5):
    #     # calculate the CI for this queue
    #     CI = confidence_interval(np.array(area_histories[i]), confidence = confidence_val)
    #     print(f"Queue {i+1} has Sample mean of the long-term average number of customers: {results[i]} with CI: {CI}")


if __name__ == "__main__":
    main()