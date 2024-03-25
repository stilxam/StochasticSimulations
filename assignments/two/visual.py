import numpy as np
import matplotlib.pyplot as plt
from Simulation import Simulation

def heatmapper(tensor):
    num_states = tensor.shape[:-1]
    num_options = tensor.shape[-1]


def line_plotter(num_its):
    simulation = Simulation(arrival_rate=0.7, departure_rates=[1, 1], m=2, theta=0.5, Max_Time=10000)

    fig, ax = plt.subplots(figsize=(80, 5))
    for i in range(num_its):
        results, _ = simulation.running_rl(alpha=0.9, epsilon=1, lr=0.2, xis=[2, 2], max_queue_length=30)

        ax.plot(results, label=f'Iteration {i}')

    ax.set(xlabel='Time', ylabel='Queue Length',
           title='Queue Length over Time')
    plt.show()

if __name__ == "__main__":
    depth = 9
    down = 5
    fig, ax = plt.subplots(nrows=down, ncols=depth, figsize=(35, 35))

    for j in range(down):
        simulation = Simulation(arrival_rate=0.7, departure_rates=[1, 1], m=2, theta=0.5, Max_Time=10**(j+1))
        res, matrix = simulation.running_rl(alpha=0.9, epsilon=1, lr=0.2, xis=[2, 2], max_queue_length=30)

        for i in range(depth):
            ax[j, i].imshow(matrix[i, :depth], cmap='viridis', interpolation='nearest')
            ax[j, i].set_title(f"T={10**(j+1)}\nLength Of Queue at Server 1= {i}")
            ax[j, i].set_xlabel('Server To Dispatch')
            ax[j, i].set_ylabel('Length of Queue at Server 2')
        # add a title to the whole figure
    fig.suptitle('Transition Matrix', fontsize=16)
    fig.colorbar(ax[0,0].imshow(matrix[0, :10], cmap='viridis', interpolation='nearest'), ax=ax, orientation='horizontal')
    plt.show()



    # the state matrix in SARSA is a representation of the learning environment
    # in the 2 server case, this matrix will be 3 dimensional, where the first two dimensions represents the corresponding
    # queue lengths at each server, while the third dimension corresponds to the weights of the different actions for the agent at that point
    # in the small multiple, column placement of the heatmap represents the length of the queue at server 1
    # inside the heatmap, the row represents the length of the queue at server 2, while the column in the heatmap represents the action to take
    # color encodes the value of the weights for the different actions
    # finally, the different rows in the small multiples represent the different times at which the simulation was run