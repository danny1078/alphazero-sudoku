
import numpy as np
import pandas as pd
from SudokuGame import SudokuGame
from Play import Play
from tqdm import tqdm
from torch.multiprocessing import Pool
def generatePuzzle(solution, num_blanks):
    # randomly mask num_blanks squares in solution with zeros
    # return solution and masked solution
    # solution is a 9x9 numpy array
    # masked_solution is a 9x9 numpy array
    gameN = 4
    masked_solution = solution.copy()
    mask = np.random.choice((gameN ** 2), num_blanks, replace=False)
    for i in mask:
        masked_solution[i // gameN][i % gameN] = 0
    return masked_solution

def string_2_array(s):
    # preprocess s to pad to the nearest square
    str1 = str(s)
    len1 = len(str1)
    while (len1 ** 0.5) % 1 != 0:
        str1 = '0' + str1
        len1 += 1

    n = int(len1 ** 0.5)
    return np.array([int(char) for char in str(str1)]).reshape(n, n)

def get_trajectories_test(net, args):
    solutions = []
    df = pd.read_csv('sudoku-4.csv')
    # hold out 100 samples for validation
    df_train = df.sample(frac=1)
    for _ in range(args['numGames']):
        solutions.append(string_2_array(df_train.sample(1)['solution'].values[0]))
    puzzles = []
    for i in range(args['numGames']):
        twodim = generatePuzzle(solutions[i], 1)
        threedim = SudokuGame.two_dim_to_three_dim(twodim)
        puzzles.append(threedim)

    puzzles = np.array(puzzles)
    target = np.array([SudokuGame.two_dim_to_three_dim(solutions[i])[1:, :, :] for i in range(args['numGames'])])
    target = target.reshape(target.shape[0], -1)
    target = target - puzzles[:, 1:, :, :].reshape(target.shape[0], -1)

    data = []
    for i in range(args['numGames']):
        data.append([puzzles[i], target[i], 1])
    return data

def get_trajectories(net, args, logger, num_blanks):
    data = []
    df = pd.read_csv('sudoku-4.csv')
    # hold out 100 samples for validation
    df_train = df.sample(frac=1)

    for _ in tqdm(range(args['numGames'])):
        solution = string_2_array(df_train.sample(1)['solution'].values[0])
        board = generatePuzzle(solution, num_blanks)
        p = Play(net, board, args)
        data += p.playGame()

    percent_perfect_games = np.mean([x[2] == 1 for x in data])
    avg_score = np.mean([x[2] for x in data])
    if logger is not None:
        logger.log({"Average score": avg_score,
                    "Percentage of perfect games": percent_perfect_games})
    return data, percent_perfect_games, avg_score

def eval_trajectories(net, args, num_blanks=12, seed=42):
    data = []
    df = pd.read_csv('sudoku-4.csv')
    random_state = np.random.RandomState(seed)
    df_train = df.sample(frac=1)
    plays = []
    for _ in range(args['numGames']):
        solution = string_2_array(df_train.sample(1, random_state=random_state)['solution'].values[0])
        board = generatePuzzle(solution, num_blanks)
        p = Play(net, board, args)
        plays.append(p)
    for i in tqdm(range(args['numGames'])):
        data += plays[i].playGame()

    percent_perfect_games = np.mean([x[2] == 1 for x in data])
    avg_score = np.mean([x[2] for x in data])

    return data, percent_perfect_games, avg_score
