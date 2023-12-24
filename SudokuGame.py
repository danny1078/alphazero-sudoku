from __future__ import print_function
import sys
sys.path.append('')
from SudokuLogic import Board
import numpy as np
from itertools import permutations

class SudokuGame():
    # square_content = {
    #     0: '-',
    #     1: 'x',
    # }

    # @staticmethod
    # def getSquarePiece(piece):
    #     return SudokuGame.square_content[piece]

    def __init__(self, n=9):
        self.n = n
        self.subgrid_size = int(n ** 0.5)

    def getN(self):
        return self.n

    def getInitBoard(self, board):
        return board

    def getBoardSize(self):
        # (a,b) tuple
        return (self.n, self.n)

    def getActionSize(self):
        # return number of actions
        return self.n**3

    def getNextState(self, board, action):
        # Action as a number from 0 to n ** 3 - 1
        # Apply action to board and return new board state
        # print("Action: ", action // (self.n ** 2), (action % (self.n ** 2)) // self.n, (action % (self.n ** 2)) % self.n + 1)
        # new_board = np.copy(board)
        # new_board[action // (self.n ** 2), (action % (self.n ** 2)) // self.n] = (action % (self.n ** 2)) % self.n + 1
        b = Board(self.n, board.copy())
        # b.board = np.copy(board)
        b.place_number(action // (self.n ** 2), (action % (self.n ** 2)) // self.n, (action % (self.n ** 2)) % self.n + 1)
        return b.board

    def getValidMoves(self, board):
        # Return a binary vector where each entry indicates if placing a number (1-9) in a cell (row, col) is valid
        b = Board(self.n)
        b.board = np.copy(board)
        valid_moves = np.zeros(self.n ** 3)
        moves = b.get_legal_moves() # list of (x, y, num) tuples
        for x, y, num in moves:
            valid_moves[x * self.n ** 2 + y * self.n + num - 1] = 1
        return valid_moves

    def getAllMoves(self, board):
        # Return a list of all moves 
        b = Board(self.n)
        b.board = np.copy(board)
        all_moves = np.zeros(self.n ** 3)
        moves = b.get_all_moves() # list of (x, y, num) tuples
        for x, y, num in moves:
            all_moves[x * self.n ** 2 + y * self.n + num - 1] = 1
        return all_moves

    def getGameEnded2(self, board):
        # Check for ongoing game
        zeros_count = np.sum(board == 0)
        if zeros_count != 0:
            return 0  # Indicates the game is still ongoing

        sum_satisfied = 0

        def check_unique(arr):
            return len(set(arr)) == self.n

        # Check satisfaction of constraints for rows and columns
        for i in range(self.n):
            if check_unique(board[i, :]):
                sum_satisfied += 1
            if check_unique(board[:, i]):
                sum_satisfied += 1

        # Check subgrids
        for i in range(0, self.n, self.subgrid_size):
            for j in range(0, self.n, self.subgrid_size):
                subgrid = board[i:i + self.subgrid_size, j:j + self.subgrid_size].flatten()
                if check_unique(subgrid):
                    sum_satisfied += 1
        if sum_satisfied == 0:
            return -1e-5
        return sum_satisfied / (3 * self.n)

    # def getCanonicalForm(self, board):
    #     return self.b.board

    def getSymmetries(self, board, pi):
        # mirror, rotational
        # TODO: add permutation of rows (with rotations = columns) and add permutation of numbers. 
        assert(len(pi) == self.getActionSize())
        pi_board = np.reshape(pi, (self.n, self.n, self.n))
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
    def two_dim_to_three_dim(board):
        n = board.shape[0]
        three_dim_board = np.zeros((n + 1, n, n))

        # Get the coordinates of non-zero elements
        x, y = np.nonzero(board)
        # One-hot encode
        three_dim_board[board[x, y], x, y] = 1

        # get the coordinates of the zero elements
        x, y = np.nonzero(board == 0)
        # One-hot encode
        three_dim_board[0, x, y] = 1

        return three_dim_board
    
    @staticmethod
    def three_dim_to_two_dim(board):
        n = board.shape[1]
        two_dim_board = np.zeros((n, n))

        # Get the coordinates of non-zero elements
        z, x, y = np.nonzero(board)
        
        # One-hot encode
        two_dim_board[x, y] = z

        return two_dim_board
    
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

    
