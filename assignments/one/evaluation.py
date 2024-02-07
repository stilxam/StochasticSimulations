from simulation import stochastic_roller_coaster
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from typing import Tuple


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


def matplotlib_plot_results(base_results: np.array, base_ci: Tuple[float, float], singles_results: np.array, singles_ci: Tuple[float,float]) -> None:
    """
    This function will plot the results of the simulations
    :param base_results: the results of the single queue system
    :param base_ci: the confidence interval of the single queue system
    :param singles_results: the results of the two queue system
    :param singles_ci: the confidence interval of the two queue system
    TODO: add the third queue system
    :return: None
    """
    max_height_frequency: float = 0.14
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharex=True, sharey=True)

    # plot the results of the single queue system
    ax[0].hist(base_results, bins=40, alpha=0.5, label='Single Queue System', color='orange',
               weights=np.ones_like(base_results) / len(base_results))
    ax[0].fill_between(base_ci, 0, max_height_frequency, color='green', alpha=0.6, label='Confidence Interval')

    # plot the results of the two queue system
    ax[1].hist(singles_results, bins=40, alpha=0.5, label='Two Queue System', color='b',
               weights=np.ones_like(singles_results) / len(singles_results))
    ax[1].fill_between(singles_ci, 0, max_height_frequency, color='green', alpha=0.6, label='Confidence Interval')

    # add legends
    ax[0].legend(loc='upper right')
    ax[1].legend(loc='upper right')

    # add labels
    ax[0].set_xlabel('Number of Groups per Boat (Single Queue System)')
    ax[1].set_xlabel('Number of Groups per Boat (Two Queue System)')
    ax[0].set_ylabel('Frequency')
    ax[1].set_ylabel('Frequency')

    # add title
    fig.suptitle(f'Queuing System Comparison n_runs={len(singles_results)}', fontsize=16)

    # set y lim
    ax[0].set_ylim(0, 0.8 * max_height_frequency)
    ax[1].set_ylim(0, 0.8 * max_height_frequency)

    # save the figure
    plt.savefig('figures/Q_systems.png')

    # show the figure
    plt.show()


def main() -> None:
    """
    This function compares the performance of the two queue systems
    :return: None
    """
    # set seeds
    np.random.seed(420)

    # set parameters
    n_runs: int = 1000000
    len_q: int = 100
    max_group_s: int = 8
    min_group_s: int = 1
    passengers: int = 8

    # run the simulations for the single queue system
    base_results = stochastic_roller_coaster(
        q_Type="BASE",
        n_runs=n_runs,
        len_q=len_q,
        max_group_s=max_group_s,
        min_group_s=min_group_s,
        passengers=max_group_s,
    )

    # calculate the confidence interval of the single queue system
    base_ci: Tuple[float, float] = confidence_interval(base_results)

    # run the simulations for the two queue system
    singles_results = stochastic_roller_coaster(
        q_Type="SINGLES",
        n_runs=n_runs,
        len_q=len_q,
        max_group_s=max_group_s,
        min_group_s=min_group_s,
        passengers=passengers,
    )

    # calculate the confidence interval of the two queue system
    single_ci: Tuple[float, float] = confidence_interval(singles_results)

    # plot the results
    matplotlib_plot_results(base_results, base_ci, singles_results, single_ci)


if __name__ == "__main__":
    main()
