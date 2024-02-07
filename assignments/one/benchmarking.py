from simulation import stochastic_roller_coaster


def confidence_interval(results)->None:
    """
    This function will calculate the confidence interval of the results
    TODO: Implement the confidence interval
    :param results:
    :return: None
    """
    mean = results.mean()
    std = results.std()
    n = len(results)
    z = 1.96 # 95% confidence interval
    lower = mean - z * (std / n**0.5)
    upper = mean + z * (std / n**0.5)
    print(f"MEAN: {mean} STD: {std}")
    print(f"CI: [{lower}, {upper}]")

def main()->None:
    """
    This function will compare the performance of the two queue systems
    :return: None
    """
    base_results = stochastic_roller_coaster(
        q_Type="BASE",
        n_runs=100000,
        len_q=100,
        max_group_s=8,
        min_group_s=1,
        passengers=8,
    )
    confidence_interval(base_results)

    singles_results = stochastic_roller_coaster(
        q_Type="SINGLES",
        n_runs=100000,
        len_q=100,
        max_group_s=8,
        min_group_s=1,
        passengers=8,
    )
    confidence_interval(singles_results)

if __name__ == "__main__":
    main()