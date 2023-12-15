
import numpy as np
import pandas as pd
from SudokuGame import SudokuGame

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
    n = int(len(str(s))**0.5)  # Determine the size of the square array
    return np.array([int(char) for char in str(s)]).reshape(n, n)

def get_trajectories(net, args):
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