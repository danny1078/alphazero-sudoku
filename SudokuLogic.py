'''
Author: Arunim
Date: Dec 1, 2023.
Board class.
Board data:
  0=empty
Squares are stored and manipulated as (x,y) tuples.
'''
import numpy as np

class Board():
    
    def __init__(self, n=9, initial_board=None):
        self.n = n
        # check that n is a square number
        assert n == int(n**0.5)**2, "n must be a square number"
        self.subgrid_size = int(n ** 0.5)

        if initial_board is None:
            self.board = np.zeros((n, n), dtype=int)
        else:
            assert isinstance(initial_board, np.ndarray), "initial_board must be a numpy ndarray"
            assert initial_board.shape == (n, n), "initial_board must have the shape (n, n)"
            self.board = initial_board

    def __getitem__(self, index):
        return self.board[index]
    
    def get_all_moves(self): 
        # any empty cell: 
        empty_cells = np.argwhere(self.board == 0)
        moves = []
        for x, y in empty_cells:
            for num in range(1, self.n + 1):
                moves.append((x, y, num))
        return moves

    def get_legal_moves(self):
        moves = []
        empty_cells = np.argwhere(self.board == 0)
        for x, y in empty_cells:
            for num in range(1, self.n + 1):
                if self.is_move_legal(x, y, num):
                    moves.append((x, y, num))
        return moves

    def has_legal_moves(self):
        return any(self.get_legal_moves())

    def is_move_legal(self, x, y, num):
        return self.is_row_valid(x, num) and \
               self.is_col_valid(y, num) and \
               self.is_subgrid_valid(x, y, num) and \
               0 < num <= self.n

    def is_row_valid(self, row, num):
        return num not in self.board[row]

    def is_col_valid(self, col, num):
        return num not in self.board[:, col]

    def is_subgrid_valid(self, x, y, num):
        startRow, startCol = self.subgrid_size * (x // self.subgrid_size), self.subgrid_size * (y // self.subgrid_size)
        return num not in self.board[startRow:startRow + self.subgrid_size, startCol:startCol + self.subgrid_size]

    def place_number(self, x, y, num):
        if self.is_move_legal(x, y, num):
            self.board[x, y] = num
            return True
        return False
    
    def is_comn(self):
        # if there are no 0's, then the board is won
        return np.all(self.board != 0)