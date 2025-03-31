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
                   exploration_weight * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6)))

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

class Cephalopod:
    def __init__(self):
        self.board = [[" ", 0]] * 25
        self.current_player = "W"
        self.last_move = None

    def somma_possibile(self, indice):
        n = indice - 5
        s = indice + 5
        e = indice + 1
        w = indice - 1

        if n < 0: # Controllo prima riga
            n = None
        if s >= 25: # Controllo ultima riga
            s = None
        if (e) % 5 == 0 or e >= 25:
            e = None
        if (w + 1) % 5 == 0:
            w = None
        somma = 0
        somma += 0 if n is None else self.board[n][1]
        somma += 0 if s is None else self.board[s][1]
        somma += 0 if e is None else self.board[e][1]
        somma += 0 if w is None else self.board[w][1]
        return somma <= 6, somma


    def get_legal_moves(self):
        ret = [[i, 0] for i in range(25) if self.board[i] == [" ", 0]]
        for i in range(25):
            possibile, somma = self.somma_possibile(i)
            somma = somma if somma <= 6 else 6
            if possibile and i not in ret:
                ret.append([i, somma])
        return ret

    def perform_move(self, move):
        new_state = Cephalopod()
        new_state.board = self.board[:]
        new_state.board[move[0]][0] = self.current_player
        new_state.board[move[0]][1] = move[1]
        new_state.current_player = "B" if self.current_player == "W" else "W"
        new_state.last_move = move
        return new_state

    def is_terminal(self):
        return self.get_winner() is not None or [" ", 0] not in self.board

    def get_winner(self):
        w = 0
        b = 0
        for i in range(25):
            if self.board[i][0] == "W":
                w += 1
            if self.board[i][0] == "B":
                b += 1
        if w == b or w + b < 25:
            return None
        return "W" if w > b else "B"

    def get_result(self):
        winner = self.get_winner()
        if winner == "W":
            return 1
        elif winner == "B":
            return -1
        return 0

    def print_board(self):
        j = 0
        line = ""
        for i in range(25):
            if j == 0:
                line += "[ "

            line += self.board[i][0] + " : " + str(self.board[i][1])
            j += 1

            if j == 5:
                line += " ]\n"
                j = 0
            else:
                line += " | "

        print(line)
def mcts(root, itermax=1000):
    """ Esegue MCTS per un numero di iterazioni """
    for _ in range(itermax):
        node = root

        # SELECTION
        while node.is_fully_expanded() and not node.state.is_terminal():
            node = node.best_child()

        # EXPANSION
        if not node.state.is_terminal():
            new_child = node.get_unvisited_child()
            if new_child:
                node = new_child

        # SIMULATION
        rollout_state = node.state
        while not rollout_state.is_terminal():
            rollout_state = rollout_state.perform_move(random.choice(rollout_state.get_legal_moves()))

        # BACKPROPAGATION
        result = rollout_state.get_result()
        while node is not None:
            node.visits += 1
            if node.state.current_player == "W":  # Se il nodo è di "X", vogliamo massimizzare il punteggio
                node.value += result
            else:
                node.value -= result
            node = node.parent

    return max(root.children, key=lambda child: child.visits)
    #or ,alternatively, return the following that considers the mean rewards over the visits, instead of the visits only
    #return root.best_child(exploration_weight=0)

def print_mcts_stats(root):
    """ Stampa statistiche su tutte le mosse possibili dal nodo root """
    print("\n📊 **Statistiche Monte Carlo Tree Search (MCTS)** 📊")
    print("Mossa | Visite | Valore Medio Vittoria (%)")
    print("---------------------------------------")
    for child in root.children:
        avg_value = round(100 * (child.value / (child.visits + 1e-6)), 2)
        print(f" {child.state.last_move}   |  {child.visits}   |  {avg_value}%")

# Simulazione di una partita Tic-Tac-Toe con MCTS
game = Cephalopod()
root = Node(game)

while not game.is_terminal():
    game.print_board()
    print_mcts_stats(root)  # Mostra le statistiche MCTS
    print("\nThinking...\n")

    root = mcts(root, itermax=500)  # Esegue MCTS per scegliere la miglior mossa
    game = root.state  # Passa al nuovo stato dopo la mossa scelta

game.print_board()
winner = game.get_winner()
if winner:
    print(f"🏆 Winner: {winner}")
else:
    print("⚖️ Draw!")
