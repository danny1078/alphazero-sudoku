from MCTS import MCTS
from utils import *
from NeuralNet import NeuralNet
from SudokuGame import SudokuGame
import numpy as np
import torch
import os
import pandas as pd

def string_2_array(s):
            n = int(len(s)**0.5)  # Determine the size of the square array
            return np.array([int(char) for char in s]).reshape(n, n)

class Play(): 
    def __init__(self, game, nnet, args, inboard=None):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.inboard = inboard

    def playGame(self):
        augment = True
        data = []
        board = self.inboard

        while True:
            temp = 0.5
            pi = self.mcts.getActionProb(board, temp=temp)
            if augment:
                sym = self.game.getSymmetries(board, pi) 
                for b,p in sym:
                    data.append([b, p, None])
            
            action = np.random.choice(len(pi), p=pi)
            
            data.append([board, pi, None])

            board = self.game.getNextState(board, action)

            end = self.game.getGameEnded2(board)
            if end != 0:
                return [(x[0], x[1], end) for x in data]

    def playGames(self, num):
        # play n games and return data in the form of examples
        examples = []
        for i in range(num):
            examples += self.playGame()
        return examples
