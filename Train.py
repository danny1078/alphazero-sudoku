import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import NNet
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
        self.df = pd.read_csv('sudoku-4.csv')
        self.game = game
        self.nnet = nnet
        self.args = args
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.data = []
        self.numGames = args.numGames
        self.min_squares = 6
        self.num_blanks = np.random.randint(1, (self.game.getN() ** 2) - self.min_squares, size=self.numGames)

    def generatePuzzle(self, solution, num_blanks):
        # randomly mask num_blanks squares in solution with zeros
        # return solution and masked solution
        # solution is a 9x9 numpy array
        # masked_solution is a 9x9 numpy array
        masked_solution = solution.copy()
        mask = np.random.choice((self.game.getN() ** 2), num_blanks, replace=False)
        for i in mask:
            masked_solution[i // self.game.getN()][i % self.game.getN()] = 0
        return masked_solution

    def learn(self, model):
        logger = wandb.init(project="alphazero_sudoku")

        for idx1 in tqdm(range(self.args.numIters)):
            print('------ITER ' + str(idx1 + 1) + '------')
            # randomly sample self.numGames boards from self.df
            plays = []

            for idx2 in range(self.numGames):
                solution = string_2_array(self.df.sample(1)['solution'].values[0])
                #puzzle = self.generatePuzzle(solution, int(np.floor(i * avg_step) + 1))
                #puzzle = self.generatePuzzle(solution, int(np.floor(0.5 * idx1+1)))
                puzzle = self.generatePuzzle(solution, 1)
                plays.append(Play(self.game, self.nnet, self.args, inboard=puzzle))

            with Pool(processes=32) as p:
                listt = [(p, model) for p in plays]
                data = p.map(playGame, listt)
                flat_data = [item for sublist in data for item in sublist]
            self.data = flat_data
            avg_completion = np.mean([x[2] for x in self.data])
            print('Average Score: ' + str(avg_completion))
            logger.log({"Average score": avg_completion,
                        "Percentage of perfect games": np.mean([x[2] == 1 + 1e-5 for x in self.data])})
            # shuffle examples before training
            print('percentage of perfect games: ' + str(np.mean([x[2] == 1 + 1e-5 for x in self.data])))
            shuffle(self.data)
            values = np.array([x[2] for x in self.data])
            print("mean and std of values: ", np.mean(values), np.std(values))

            self.nnet.try_train(model, self.data, wandb_logger=logger)


def playGame(tup):
    p, model = tup
    data = p.playGame(model)
    return data
