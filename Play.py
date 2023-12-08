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
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)

        df = pd.read_csv('sudoku-small.csv')
        

        self.inboard = string_2_array(df.iloc[3]['puzzle'])
        self.solution = string_2_array(df.iloc[3]['solution'])
    
    def playGame(self):
        augment = False
        data = []
        board = self.game.getInitBoard(self.inboard, self.solution)

        while True:
            # print(self.board)
            canonicalBoard = self.game.getCanonicalForm(board)
            temp = int(1e-3)
            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            
            if augment: 
                sym = self.game.getSymmetries(canonicalBoard, pi) 
                for b,p in sym:
                    data.append([b, p, None])
            
            action = np.random.choice(len(pi), p=pi)
            
            board = self.game.getNextState(board, action)

            data.append([canonicalBoard, pi, None])

            end = self.game.getGameEnded(board)
            if end != 0:
                return [(x[0], x[1], end) for x in data]

    def playGames(self, num):
        print(num)
        # play n games and return data in the form of examples
        examples = []
        for i in range(num):
            examples += self.playGame()
        return examples