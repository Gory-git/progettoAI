import math
import random
import algoritmi
from TicTacToe import TicTacToe
from Game import play_game


def random_player(game, state): return random.choice(list(game.actions(state)))

def player(search_algorithm):
    """A game player who uses the specified search algorithm"""
    return lambda game, state: search_algorithm(game, state)[1]


var = play_game(TicTacToe(), dict(X=player(algoritmi.minimax_search), O=player(algoritmi.alphabeta_search)), verbose=True).utility

# Simulazione di una partita con MCTS
# game = Cephalopod() # TODO
# root = Node(game)
#
# while not game.is_terminal():
#     game.print_board()
#     print_mcts_stats(root)  # Mostra le statistiche MCTS
#     print("\nThinking...\n")
#
#     root = mcts(root, itermax=500)  # Esegue MCTS per scegliere la miglior mossa
#     game = root.state  # Passa al nuovo stato dopo la mossa scelta
#
# game.print_board()
# winner = game.get_winner()
# if winner:
#     print(f"🏆 Winner: {winner}")
# else:
#     print("⚖️ Draw!")