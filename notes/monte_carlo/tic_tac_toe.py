import numpy as np
import random
import time
from joblib import Parallel, delayed
import multiprocessing as mp
from numpy import array, arange, zeros
from typing import Callable


def is_winner(board: array, player):
    output = False
    for i in range(3):
        if all(board[i, :] == player) or all(board[:, i] == player):
            output = True
            break

    diag: array = np.diag(board)
    flip_diag: array = np.diag(np.fliplr(board))
    if all(diag == player) or all(flip_diag == player):
        output = True

    # if output:
        # print(f"Player {player} wins")
        # print(board)

    return output


def update_board(board, x, y, player):
    board[x, y] = player
    return board


def random_choose_move(board, player):
    possible_moves = np.argwhere(board == 0)
    if len(possible_moves) == 0:
        return None
    else:
        move = random.choice(possible_moves)
        return move


def smart_choose_move(board, player):
    possible_moves = np.argwhere(board == 0)

    if len(possible_moves) == 0:
        return None
    else:
        for move in possible_moves:
            board = update_board(board, move[0], move[1], player)
            if is_winner(board, player):
                return move
            else:
                board = update_board(board, move[0], move[1], 0)
        return random_choose_move(board, player)


def play_game(player_one_brain: Callable = random_choose_move, player_two_brain: Callable = random_choose_move) -> int:
    board = np.zeros((3, 3))
    for i in range(10):
        player = 1 if i % 2 == 0 else 2
        if player == 1:
            move = player_one_brain(board, player)
        else:
            move = player_two_brain(board, player)

        if move is not None:
            board = update_board(board, move[0], move[1], player)

            if is_winner(board, player):
                return player
        else:
            # print("No more moves, game is a draw")
            return -1


def experiment(num_trials: int, player_one_brain: Callable = random_choose_move, player_two_brain: Callable = random_choose_move) -> array:
    results = zeros(num_trials)
    for i in range(num_trials):
        results[i] = play_game(player_one_brain, player_two_brain)
    return results

def parallelized_experiment(n_trials, brain_one, brain_two, num_cpu = mp.cpu_count()-1)->array:
    results = Parallel(n_jobs=num_cpu)(delayed(play_game)(brain_one, brain_two) for _ in range(n_trials))
    return np.array(results)




if __name__ == "__main__":
    num_trials = 100000
    t_init = time.time()
    results = experiment(num_trials, smart_choose_move, random_choose_move)
    t_term = time.time()

    print(f"TIME FOR SINGLE CORE: {t_term - t_init}")
    print(f"Average number of wins for player 1: {(results==1).mean()}")
    print(f"Average number of wins for player 2: {(results==2).mean()}")
    print(f"Average number of draws: {(results==-1).mean()}")

    t_init = time.time()
    results = parallelized_experiment(num_trials, random_choose_move, smart_choose_move)
    t_term = time.time()

    print(f"TIME FOR MULTITHREADED : {t_term - t_init}")
    print(f"Average number of wins for player 1: {(results==1).mean()}")
    print(f"Average number of wins for player 2: {(results==2).mean()}")
    print(f"Average number of draws: {(results==-1).mean()}")



