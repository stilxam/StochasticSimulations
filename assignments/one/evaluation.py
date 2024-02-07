from simulation import stochastic_roller_coaster
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from typing import Tuple


def confidence_interval(results: np.array, confidence: float = 0.05) -> Tuple[float, float]:
    """
    This function will calculate the confidence interval of the results
    :param results: the results of the simulations
    :param confidence: the confidence level
    :return: None
    """
    mean = results.mean()
    var = results.var()
    n = len(results)
    z = stats.norm.ppf(1 - (confidence) / 2)
    half_width = z * (var / n) ** 0.5
    lower = mean - half_width
    upper = mean + half_width

    print(f"Group Per Boat \nMEAN: {mean} \nVAR: {var}\nSTD {var ** 0.5}\nN: {n}\nZ: {z}\nHalf Width: {half_width}")
    print(f"CI: [{lower}, {upper}]")
    return lower, upper


def plot_results(base_results, singles_results) -> None:
    """
    This function will plot the results of the simulations
    :param base_results:
    :param singles_results:
    :return: None
    """
    # declare a figure

    fig = go.Figure(
        layout=go.Layout(
            title=go.layout.Title(text="Queue System Comparison")
        )
    )
    fig.add_histogram(x=base_results, name="Single Queue System")
    fig.add_histogram(x=singles_results, name="Two Queue System")
    fig.show()


def matplotlib_plot_results(base_results, base_ci , singles_results, singles_ci) -> None:
    """
    This function will plot the results of the simulations
    :param base_results:
    :param singles_results:
    :return: None
    """
    max_height_frequency = 0.14
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharex=True, sharey=True)
    ax[0].hist(base_results, bins=40, alpha=0.5, label='Single Queue System', color='orange',
            weights=np.ones_like(base_results) / len(base_results))
    ax[0].fill_between(base_ci, 0, max_height_frequency, color='green', alpha=0.6, label='Confidence Interval')

    ax[1].hist(singles_results, bins=40, alpha=0.5, label='Two Queue System', color='b',
            weights=np.ones_like(singles_results) / len(singles_results))
    ax[1].fill_between(singles_ci, 0, max_height_frequency, color='green', alpha=0.6, label='Confidence Interval')
    ax[0].legend(loc='upper right')
    ax[1].legend(loc='upper right')
    ax[0].set_xlabel('Number of Groups per Boat (Single Queue System)')
    ax[1].set_xlabel('Number of Groups per Boat (Two Queue System)')
    ax[0].set_ylabel('Frequency')
    fig.suptitle('Queuing System Comparison')

    # set y lim
    ax[0].set_ylim(0, 0.8*max_height_frequency)
    ax[1].set_ylim(0, 0.8*max_height_frequency)
    plt.savefig('figures/Q_systems.png')
    plt.show()


def main() -> None:
    """
    This function compares the performance of the two queue systems
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
    base_ci : Tuple[float,float] = confidence_interval(base_results)

    singles_results = stochastic_roller_coaster(
        q_Type="SINGLES",
        n_runs=100000,
        len_q=100,
        max_group_s=8,
        min_group_s=1,
        passengers=8,
    )
    single_ci: Tuple[float,float] = confidence_interval(singles_results)

    # plot_results(base_results, singles_results)
    matplotlib_plot_results(base_results, base_ci, singles_results, single_ci)


if __name__ == "__main__":
    main()
