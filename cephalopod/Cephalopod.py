from Game import Game
from Board import Board

class Cephalopod(Game):
    def __init__(self, height= 5, width= 5):
        self.initial = Board(height=height, width=width, to_move='W', utility=0)
        self.squares = {(x, y) for x in range(width) for y in range(height)}

    def is_terminal(self, board):
        return board.utility != 0 or len(self.squares) == len(board)

    def utility(self, board, player):
        return board.utility if player == 'W' else -board.utility

    def result(self, board, square):
        player = board.to_move
        

