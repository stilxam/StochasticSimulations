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
    lambda_param = 0.7
    mus = [1]
    thetas = [0.6]
    m = 1
    its = 10000
    Max_time = 10000

    t1 = time.time()

    results = [[] for _ in range(len(mus))]

    # -----------------------------------------------Random Dispatcher results----------------------------------------------- #

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
        results = sim.perform_n_simulations(its)

        results = np.array(results)
        mean = results.mean(axis=0)

        # calculate CI
        low_bound, high_bound = confidence_interval(results=results)

        print(f"--------------Simulation results (theta = {theta})--------------")
        for i in range(m):
            print(
                f"Queue {i + 1} has Sample mean of the long-term average number of customers: {mean[i]} with CI: [{low_bound[i]}, {high_bound[i]}]")
        
        print("\n")

    # -----------------------------------------------Sarsa Dispatcher results----------------------------------------------- #
    
    # # Define parameters
    # m_sarsa = 2
    # arrival_rate = 0.7
    # departure_rates = [1, 1]
    # theta = 0.5
    # Max_Time = 10000
    # alpha = 0.9
    # epsilon = 1
    # lr = 0.2
    # xis = [2, 2]
    # max_queue_length = 30
    # n_its = 10000 

    # # Initialize arrays to store results
    # results = np.empty((n_its, m_sarsa))
    # q_s = np.empty((n_its, max_queue_length, max_queue_length, m_sarsa+1))

    # # Initialize simulation (again)
    # sim2 = Simulation(lambda_param, mus, m_sarsa, theta, Max_time)

    # # Perform simulations
    # for i in range(n_its):
    #     results[i], q_s[i] = sim2.simulate_sarsa_dispatcher(alpha=alpha, epsilon=epsilon, lr=lr, xis=xis, max_queue_length=max_queue_length)

    # # Print results
    # print(f"mean: {results.mean(axis=0)}")
    # print(f"std: {results.std(axis=0)}")
    # print(results)
    # print(f"average q_s: {q_s[0]}")

    



    t2 = time.time()
    print(f"time: {t2 - t1}")


if __name__ == "__main__":
    main()