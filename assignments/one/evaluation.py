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

    print(f"MEAN: {mean} \nVAR: {var}\nSTD {var ** 0.5}\nN: {n}\nZ: {z}\nHalf Width: {half_width}")
    print(f"CI: [{lower}, {upper}]")
    return lower, upper


def matplotlib_plot_results(
        title: str,
        base_results: np.array,
        base_ci: Tuple[float, float],
        singles_results: np.array,
        singles_ci: Tuple[float, float],
        dynamic_results: np.array,
        dynamic_ci: Tuple[float, float]
):
    """
    This function will plot the results of the simulations
    :param base_results: the results of the single queue system
    :param base_ci: the confidence interval of the single queue system
    :param singles_results: the results of the two queue system
    :param singles_ci: the confidence interval of the two queue system
    :param dynamic_results: the results of the two queue system
    :param dynamic_ci: the confidence interval of the two queue system
    :return: None
    """
    max_height_frequency: float = 0.2
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharex=True, sharey=True)

    # plot the results of the single queue system
    ax[0].hist(base_results, bins=40, alpha=0.5, label='Single Queue System', color='orange',
               weights=np.ones_like(base_results) / len(base_results))
    ax[0].fill_between(base_ci, 0, max_height_frequency, color='green', alpha=0.6, label='Confidence Interval')

    # plot the results of the two queue system
    ax[1].hist(singles_results, bins=40, alpha=0.5, label='Two Queue System', color='b',
               weights=np.ones_like(singles_results) / len(singles_results))
    ax[1].fill_between(singles_ci, 0, max_height_frequency, color='green', alpha=0.6, label='Confidence Interval')

    # plot the results of the dynamic queue system
    ax[2].hist(dynamic_results, bins=40, alpha=0.5, label='Dynamic Queue System', color='b',
               weights=np.ones_like(dynamic_results) / len(dynamic_results))
    ax[2].fill_between(dynamic_ci, 0, max_height_frequency, color='green', alpha=0.6, label='Confidence Interval')

    # add legends
    ax[0].legend(loc='upper right')
    ax[1].legend(loc='upper right')
    ax[2].legend(loc='upper right')

    # add labels
    ax[0].set_xlabel(f'Number of {title}(Single Queue System)')
    ax[1].set_xlabel(f'Number of {title}(Two Queue System)')
    ax[2].set_xlabel(f'Number of {title}(Dymanic Queue System)')
    ax[0].set_ylabel('Frequency')
    ax[1].set_ylabel('Frequency')
    ax[2].set_ylabel('Frequency')

    # add title
    fig.suptitle(f'Queuing System Comparison n_runs={len(singles_results)}', fontsize=16)

    # set y lim
    ax[0].set_ylim(0, 0.8 * max_height_frequency)
    ax[1].set_ylim(0, 0.8 * max_height_frequency)

    # save the figure
    # plt.savefig(f'figures/{title}_Q_systems.png')

    # show the figure
    return plt


def queue_plot_results(
        title: str,
        base_results: np.array,
        base_ci: Tuple[float, float],
        singles_t_results: np.array,
        singles_t_ci: Tuple[float, float],
        singles_s_results: np.array,
        singles_s_ci: Tuple[float, float],
        singles_r_results: np.array,
        singles_r_ci: Tuple[float, float],
        dynamic_results: np.array,
        dynamic_ci: Tuple[float, float]
):
    """
    This function will plot the results of the simulations
    :param singles_r_ci:
    :param singles_r_results:
    :param singles_s_results:
    :param singles_t_ci:
    :param singles_t_results:
    :param singles_s_ci:
    :param base_results: the results of the single queue system
    :param base_ci: the confidence interval of the single queue system
    :param singles_results: the results of the two queue system
    :param singles_ci: the confidence interval of the two queue system
    :param dynamic_results: the results of the two queue system
    :param dynamic_ci: the confidence interval of the two queue system
    :return: None
    """
    max_height_frequency: float = 0.2
    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(30, 5),  sharey=True)

    # plot the results of the single queue system
    ax[0].hist(base_results, bins=40, alpha=0.5, label='Single Queue System', color='orange',
               weights=np.ones_like(base_results) / len(base_results))
    ax[0].fill_between(base_ci, 0, max_height_frequency, color='green', alpha=0.6, label='Confidence Interval')

    # plot the results of the two queue system
    ax[1].hist(singles_t_results, bins=40, alpha=0.5, label='Total Two Queue System', color='b',
               weights=np.ones_like(singles_t_results) / len(singles_t_results))
    ax[1].fill_between(singles_t_ci, 0, max_height_frequency, color='green', alpha=0.6, label='Confidence Interval')

    ax[2].hist(singles_s_results, bins=40, alpha=0.5, label='Single Two Queue System', color='b',
               weights=np.ones_like(singles_s_results) / len(singles_s_results))
    ax[2].fill_between(singles_s_ci, 0, max_height_frequency, color='green', alpha=0.6, label='Confidence Interval')

    ax[3].hist(singles_r_results, bins=40, alpha=0.5, label='Regular Two Queue System', color='b',
               weights=np.ones_like(singles_r_results) / len(singles_r_results))
    ax[3].fill_between(singles_r_ci, 0, max_height_frequency, color='green', alpha=0.6, label='Confidence Interval')

    # plot the results of the dynamic queue system
    ax[4].hist(dynamic_results, bins=40, alpha=0.5, label='Dynamic Queue System', color='b',
               weights=np.ones_like(dynamic_results) / len(dynamic_results))
    ax[4].fill_between(dynamic_ci, 0, max_height_frequency, color='green', alpha=0.6, label='Confidence Interval')

    # add legends
    ax[0].legend(loc='upper right')
    ax[1].legend(loc='upper right')
    ax[2].legend(loc='upper right')
    ax[3].legend(loc='upper right')
    ax[4].legend(loc='upper right')

    # add labels
    ax[0].set_xlabel(f'Mean {title}(Single Queue System)')
    ax[1].set_xlabel(f'Mean {title}(Total Two Queue System)')
    ax[2].set_xlabel(f'Mean {title}(Regular Two Queue System)')
    ax[3].set_xlabel(f'Mean {title}(Single Two Queue System)')
    ax[4].set_xlabel(f'Mean {title}(Dymanic Queue System)')
    ax[0].set_ylabel('Frequency')
    ax[1].set_ylabel('Frequency')
    ax[2].set_ylabel('Frequency')
    ax[3].set_ylabel('Frequency')
    ax[4].set_ylabel('Frequency')

    # add title
    fig.suptitle(f'Queuing System Comparison n_runs={len(base_results)}', fontsize=16)

    # set y lim
    ax[0].set_ylim(0, 0.8 * max_height_frequency)
    ax[1].set_ylim(0, 0.8 * max_height_frequency)

    # save the figure
    # plt.savefig(f'figures/{title}_Q_systems.png')

    # show the figure
    return plt
