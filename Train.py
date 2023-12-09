import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

from MCTS import MCTS
from Play import Play

class Train(): 

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()
        self.data = []

    def getData(self):
        # use args to import data of already-played-games / data from self-play
        # update self.data
        pass

    def learn(self):
        for i in tqdm(range(self.args.numIters)):
            print('------ITER ' + str(i+1) + '------')
            
            p = Play(self.game, self.nnet, self.args)
            data = p.playGames(num=self.args.numEps)
            self.data = data

            # shuffle examples before training
            shuffle(self.data)
            self.nnet.train(self.data)
            
        pass
