import math
import random

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_legal_moves())

    def best_child(self, exploration_weight=1.41):
        """ Seleziona il miglior figlio usando UCT (Upper Confidence Bound for Trees) """
        return max(self.children, key=lambda child: (child.value / (child.visits + 1e-6)) +
                                                    exploration_weight * math.sqrt(
            math.log(self.visits + 1) / (child.visits + 1e-6)))

    def get_unvisited_child(self):
        """ Ritorna un figlio non ancora esplorato, se esiste """
        unvisited_moves = [move for move in self.state.get_legal_moves()
                           if move not in [child.state.last_move for child in self.children]]
        if unvisited_moves:
            move = random.choice(unvisited_moves)
            new_state = self.state.perform_move(move)
            new_child = Node(new_state, parent=self)
            self.children.append(new_child)
            return new_child
        return None

