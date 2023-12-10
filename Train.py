import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

from MCTS import MCTS
from Play import Play, string_2_array
import pandas as pd
from multiprocessing import Pool
import wandb
from SudokuGame import SudokuGame


class Train():

    def __init__(self, game, nnet, args):
        self.df = pd.read_csv('sudoku.csv')
        self.game = game
        self.nnet = nnet
        self.args = args
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.data = []
        self.numGames = args.numGames

    def generatePuzzle(self, solution, num_blanks):
        # randomly mask num_blanks squares in solution with zeros
        # return solution and masked solution
        # solution is a 9x9 numpy array
        # masked_solution is a 9x9 numpy array
        masked_solution = solution.copy()
        mask = np.random.choice(81, num_blanks, replace=False)
        for i in mask:
            masked_solution[i // 9][i % 9] = 0
        return masked_solution

    def learn(self, min_squares=20):
        logger = wandb.init(project="alphazero_sudoku")
        avg_step = (81 - min_squares) / self.args.numIters
        for idx1 in tqdm(range(self.args.numIters)):
            print('------ITER ' + str(idx1 + 1) + '------')
            # randomly sample self.numGames boards from self.df
            plays = []
            num_blanks = np.random.randint(1, 81 - min_squares, size=self.numGames)
            for idx2 in range(self.numGames):
                solution = string_2_array(self.df.sample(1)['solution'].values[0])
                #puzzle = self.generatePuzzle(solution, int(np.floor(i * avg_step) + 1))
                puzzle = self.generatePuzzle(solution, num_blanks[idx2])
                plays.append(Play(self.game, self.nnet, self.args, inboard=puzzle))

            with Pool(processes=32) as p:
                data = p.map(playGame, plays)
                flat_data = [item for sublist in data for item in sublist]
            self.data = flat_data
            avg_completion = np.mean([x[2] for x in self.data])
            print('Average Score: ' + str(avg_completion))
            logger.log({"Average score": avg_completion,
                       "Number of blanks": num_blanks.mean()})
            # shuffle examples before training
            shuffle(self.data)
            self.nnet.train(self.data, wandb_logger=logger)


def playGame(p):
    data = p.playGame()
    return data
