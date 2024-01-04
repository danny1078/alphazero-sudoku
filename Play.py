from SudokuGame import SudokuGame
from MCTS import MCTS
import numpy as np
class Play:
    def __init__(self, net, init_board, args):
        self.net = net
        self.init_board = init_board
        self.game = SudokuGame(args['board_size'])
        self.args = args
        self.mcts = MCTS(self.game, self.net, self.args)

    def playGame(self):
        data = []
        board = self.init_board

        while True:
            pi = self.mcts.getActionProb(board, temp=self.args['temp'])

            if self.args['augment']:
                sym = self.game.getSymmetries(board, pi)
                for b, p in sym:
                    data.append([SudokuGame.two_dim_to_three_dim(b), p, None])
            action = np.random.choice(len(pi), p=pi)

            data.append([SudokuGame.two_dim_to_three_dim(board), pi, None])
            #SudokuGame.display(board)
            #print(pi)
            board = self.game.getNextState(board, action)
            #SudokuGame.display(board)
            end = self.game.getGameEnded2(board)
            if end != 0:
                return [(x[0], x[1], end) for x in data]
