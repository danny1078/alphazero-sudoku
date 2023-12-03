from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .SudokuLogic import Board
import numpy as np
from itertools import permutations

class SudokuGame(Game):
    # square_content = {
    #     0: '-',
    #     1: 'x',
    # }

    # @staticmethod
    # def getSquarePiece(piece):
    #     return SudokuGame.square_content[piece]

    def __init__(self, n=9):
        self.n = n

    def getInitBoard(self):
        # Initialize an empty Sudoku board
        b = Board(self.n)
        return b.board # np array (n, n)

    def getBoardSize(self):
        # (a,b) tuple
        return (self.n, self.n)

    def getActionSize(self):
        # return number of actions
        return self.n**3

    def getNextState(self, board, action):
        # Action as (row, col, number) to place on the board
        # Apply action to board and return new board state
        new_board = np.copy(board)
        new_board[action[0], action[1]] = action[2]
        return new_board

    def getValidMoves(self, board):
        # Return a binary vector where each entry indicates if placing a number (1-9) in a cell (row, col) is valid
        valid_moves = np.zeros((9, 9, 9))
        moves = board.get_legal_moves() # list of (x, y, num) tuples
        for x, y, num in moves:
            valid_moves[x, y, num-1] = 1
        return valid_moves

    def getAllMoves(self, board):
        # Return a list of all moves 
        all_moves = np.zeros((9, 9, 9))
        moves = board.get_all_moves() # list of (x, y, num) tuples
        for x, y, num in moves:
            all_moves[x, y, num-1] = 1
        return all_moves

    def getGameEnded(self, board, solution):
        three_dim_board = self.two_dim_to_three_dim(board)
        # If there are no zeros on the board, the game is over
        if not np.all(board != 0):
            return 0
        else:
            # take the inner product of the board and the solution to get the number of errors
            elementwise = three_dim_board * solution
            summed_array = np.sum(elementwise, axis=2) # sum along the third dimension
            errors = np.size(summed_array) - np.sum(summed_array)
            if errors == 0:
                return 1e3
            else:
                return -(errors ** 2)

    def getCanonicalForm(self, board):
        return board

    def getSymmetries(self, board, pi):
        # mirror, rotational
        # TODO: add permutation of rows (with rotations = columns) and add permutation of numbers. 
        assert(len(pi) == self.getActionSize())
        pi_board = np.reshape(pi, (self.n, self.n))
        l = []
        small_perms = list(permutations(range(int(self.n**0.5))))
        big_perms = list(permutations(range(self.n))) # 9! = 362880 !!!!, randomly sample these ones, or don't at all. prob better to just build this invariance in
        
        for outer_perm in small_perms: 
            for i in range(1, 5):
                for j in [True, False]: 
                    newB = np.copy(board)
                    newPi = np.copy(pi_board)
                    
                    # permute n**0.5 x n rows 
                    for row in range(self.n):
                        newB[row] = board[(outer_perm[row // int(self.n**0.5)]) * int(self.n**0.5) + row % int(self.n**0.5)]
                        newPi[row] = pi_board[(outer_perm[row // int(self.n**0.5)]) * int(self.n**0.5) + row % int(self.n**0.5)]

                    newB = np.rot90(board, i)
                    newPi = np.rot90(pi_board, i)
                    if j: # note: LR flips combined with rotations can = UD flips
                        newB = np.fliplr(newB)
                        newPi = np.fliplr(newPi)
                    l += [(newB, list(newPi.ravel()))]
        return l

    def stringRepresentation(self, board):
        return "".join(str(square) for row in board for square in row)

    def getScore(self, board, solution):
        three_dim_board = self.two_dim_to_three_dim(board)
        elementwise = three_dim_board * solution
        summed_array = np.sum(elementwise, axis=2) # sum along the third dimension
        errors = np.size(summed_array) - np.sum(summed_array)
        if errors == 0:
            return 1e3
        else:
            return -(errors ** 2)

    @staticmethod
    def display(board):
        n = board.shape[0]
        print("   ", end="")
        for y in range(n):
            print(y, end=" ")
        print("\n" + "-" * (2 * n + 3))
        for y in range(n):
            print(y, "|", end="")  # print the row number
            for x in range(n):
                piece = board[y][x]  # get the piece to print
                print(piece if piece != 0 else ".", end=" ")
            print("|")
        print("-" * (2 * n + 3))

    
