from Simulation import Simulation
import numpy as np
import time


def confidence_interval(results, confidence=0.95):
    results = np.array(results)
    mean = results.mean(axis=0)
    std_dev = results.std(axis=0, ddof=1)

    low = mean - (1.96 * (std_dev / np.sqrt(len(results))))
    high = mean + (1.96 * (std_dev / np.sqrt(len(results))))

    return low, high


def main():
    np.random.seed(42069)
    lambda_param = 3
    mus = [4, 5, 6, 7, 8]
    thetas = [0.60, 0.85]
    m = 5
    its = 10000
    Max_time = 10000

    t1 = time.time()

    results = [[] for _ in range(len(mus))]

    print("--------------Setup:--------------")
    print(f"Random Seed: {np.random.seed(42069)}")
    print(f"Lambda Parameter: {lambda_param}")
    print(f"Mus: {mus}")
    print(f"Thetas: {thetas}")
    print(f"m: {m}")
    print(f"Iterations: {its}")
    print(f"Max Time: {Max_time}")
    print(f"--------------End Setup--------------")
    print("\n")

    for theta in thetas:

        sim = Simulation(lambda_param, mus, m, theta, Max_time)
        results = sim.perform_n_simulations(its, dispatcher = 0)

        results = np.array(results)
        mean = results.mean(axis=0)

        # calculate CI
        low_bound, high_bound = confidence_interval(results=results)

        print(f"--------------Simulation results (theta = {theta})--------------")
        for i in range(m):
            print(
                f"Queue {i + 1} has Sample mean of the long-term average number of customers: {mean[i]} with CI: [{low_bound[i]}, {high_bound[i]}]")
        
        print("\n")


    t2 = time.time()
    print(f"time: {t2 - t1}")


if __name__ == "__main__":
    main()