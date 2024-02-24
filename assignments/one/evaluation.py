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
    :param base_results: the results of the Model without single-rider queue
    :param base_ci: the confidence interval of the Model without single-rider queue
    :param singles_results: the results of the two queue system
    :param singles_ci: the confidence interval of the two queue system
    :param dynamic_results: the results of the two queue system
    :param dynamic_ci: the confidence interval of the dynamic queue system
    :return: None
    """
    queue_systems = [
    {"results": base_results, "ci": base_ci, "label": 'Model without single-rider queue', "color": 'orange'},
    {"results": singles_results, "ci": singles_ci, "label": 'Model with single rider queue', "color": 'b'},
    {"results": dynamic_results, "ci": dynamic_ci, "label": 'Model with Dynamic queue', "color": 'purple'}
    ]

    for i, queue_system in enumerate(queue_systems):
        fig, ax = plt.subplots(figsize=(5, 5))

        # plot the results of the queue system
        n, bins, patches = ax.hist(queue_system["results"], bins=40, alpha=0.5, label=queue_system["label"], color=queue_system["color"],
            weights=np.ones_like(queue_system["results"]) / len(queue_system["results"]))
    
        # calculate the maximum frequency
        max_height_frequency = max(n)

        ax.fill_between(queue_system["ci"], 0, max_height_frequency, color='green', alpha=0.6, label='Confidence Interval')

        # add legend
        ax.legend(loc='upper right')

        # add labels
        ax.set_xlabel(f'Mean boat occupancy')
        ax.set_ylabel('Frequency')

        # add title
        fig.suptitle(f'{queue_system["label"]}\nn_runs={len(queue_system["results"])}', fontsize=16)

        # set y lim
        # ax.set_ylim(0, 0.8 * max_height_frequency)

        # save the figure
        plt.savefig(f'FIGURES/{title}_Q_system_{i}.png')

        # close the figure
        plt.close(fig)
        
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
    :param base_results: the results of the Model without single-rider queue
    :param base_ci: the confidence interval of the Model without single-rider queue
    :param singles_results: the results of the two queue system
    :param singles_ci: the confidence interval of the two queue system
    :param dynamic_results: the results of the two queue system
    :param dynamic_ci: the confidence interval of the dynamic queue system
    :return: None
    """
    queue_systems = [
    {"results": base_results, "ci": base_ci, "label": 'Model without single-rider queue', "xlabel": f'Model without single-rider queue', "color": 'orange'},
    {"results": singles_t_results, "ci": singles_t_ci, "label": 'Model with single rider queue (Total)', "xlabel": f'Model with single rider queue (Total)', "color": 'b'},
    {"results": singles_s_results, "ci": singles_s_ci, "label": 'Model with single rider queue (singles)', "xlabel": f'Model with single rider queue (Singles)', "color": 'b'},
    {"results": singles_r_results, "ci": singles_r_ci, "label": 'Model with single rider queue (regular)', "xlabel": f'Model with single rider queue (Regular)', "color": 'b'},
    {"results": dynamic_results, "ci": dynamic_ci, "label": 'Dynamic Queue System', "xlabel": f'Dymanic Queue System', "color": 'purple'}
    ]

    for i, queue_system in enumerate(queue_systems):
        fig, ax = plt.subplots(figsize=(5, 5))

        # plot the results of the queue system
        n, bins, patches = ax.hist(queue_system["results"], bins=40, alpha=0.5, label=queue_system["label"], color=queue_system["color"],
            weights=np.ones_like(queue_system["results"]) / len(queue_system["results"]))

        # calculate the maximum frequency
        max_height_frequency = max(n)

        ax.fill_between(queue_system["ci"], 0, max_height_frequency, color='green', alpha=0.6, label='Confidence Interval')

        # add legend
        ax.legend(loc='upper right')

        # add labels
        ax.set_xlabel("Mean Queue Length")
        ax.set_ylabel('Frequency')

        # add title
        fig.suptitle(f'{queue_system["xlabel"]}\nn_runs = {len(queue_system["results"])}', fontsize=16)

        # set y lim
        # ax.set_ylim(0, 0.8 * max_height_frequency)

        # save the figure
        plt.savefig(f'FIGURES/{title}_Q_system_{i}.png')

        # close the figure
        plt.close(fig)

    return plt
