import math
import copy
import itertools
import random
from collections import namedtuple

import numpy as np

from utils4e import vector_add, MCT_Node, ucb

def minimax_search(game, state):
    """Search game tree to determine best move; return (value, move) pair."""

    player = state.to_move

    def max_value(state):
        if game.is_terminal(state):
            return game.utility(state, player), None
        v, move = -infinity, None
        for a in game.actions(state):
            v2, _ = min_value(game.result(state, a))
            if v2 > v:
                v, move = v2, a
        return v, move

    def min_value(state):
        if game.is_terminal(state):
            return game.utility(state, player), None
        v, move = +infinity, None
        for a in game.actions(state):
            v2, _ = max_value(game.result(state, a))
            if v2 < v:
                v, move = v2, a
        return v, move

    return max_value(state)

infinity = math.inf

def alphabeta_search(game, state):
    """Search game to determine best action; use alpha-beta pruning.
    As in [Figure 5.7], this version searches all the way to the leaves."""

    player = state.to_move

    def max_value(state, alpha, beta):
        if game.is_terminal(state):
            return game.utility(state, player), None
        v, move = -infinity, None
        for a in game.actions(state):
            v2, _ = min_value(game.result(state, a), alpha, beta)
            if v2 > v:
                v, move = v2, a
                alpha = max(alpha, v)
            if v >= beta:
                return v, move
        return v, move

    def min_value(state, alpha, beta):
        if game.is_terminal(state):
            return game.utility(state, player), None
        v, move = +infinity, None
        for a in game.actions(state):
            v2, _ = max_value(game.result(state, a), alpha, beta)
            if v2 < v:
                v, move = v2, a
                beta = min(beta, v)
            if v <= alpha:
                return v, move
        return v, move

    return max_value(state, -infinity, +infinity)



def cache1(function):
    "Like lru_cache(None), but only considers the first argument of function."
    cache = {}
    def wrapped(x, *args):
        if x not in cache:
            cache[x] = function(x, *args)
        return cache[x]
    return wrapped


def alphabeta_search_tt(game, state):
    """Search game to determine best action; use alpha-beta pruning.
    As in [Figure 5.7], this version searches all the way to the leaves."""

    player = state.to_move

    @cache1
    def max_value(state, alpha, beta):
        if game.is_terminal(state):
            return game.utility(state, player), None
        v, move = -infinity, None
        for a in game.actions(state):
            v2, _ = min_value(game.result(state, a), alpha, beta)
            if v2 > v:
                v, move = v2, a
                alpha = max(alpha, v)
            if v >= beta:
                return v, move
        return v, move

    @cache1
    def min_value(state, alpha, beta):
        if game.is_terminal(state):
            return game.utility(state, player), None
        v, move = +infinity, None
        for a in game.actions(state):
            v2, _ = max_value(game.result(state, a), alpha, beta)
            if v2 < v:
                v, move = v2, a
                beta = min(beta, v)
            if v <= alpha:
                return v, move
        return v, move

    return max_value(state, -infinity, +infinity)

def cutoff_depth(d):
    """A cutoff function that searches to depth d."""
    return lambda game, state, depth: depth > d

def zero_alphabeta_search(game, state, cutoff=cutoff_depth(2)):
    """Search game to determine best action; use alpha-beta pruning.
    As in [Figure 5.7], this version searches all the way to the leaves."""

    player = state.to_move

    @cache1
    def max_value(state, alpha, beta, depth):
        if game.is_terminal(state):
            return game.utility(state, player), None
        if cutoff(game, state, depth):
            return 0, None
        v, move = -infinity, None
        for a in game.actions(state):
            v2, _ = min_value(game.result(state, a), alpha, beta, depth+1)
            if v2 > v:
                v, move = v2, a
                alpha = max(alpha, v)
            if v >= beta:
                return v, move
        return v, move

    @cache1
    def min_value(state, alpha, beta, depth):
        if game.is_terminal(state):
            return game.utility(state, player), None
        if cutoff(game, state, depth):
            return 0, None
        v, move = +infinity, None
        for a in game.actions(state):
            v2, _ = max_value(game.result(state, a), alpha, beta, depth + 1)
            if v2 < v:
                v, move = v2, a
                beta = min(beta, v)
            if v <= alpha:
                return v, move
        return v, move

    return max_value(state, -infinity, +infinity, 0)

def h_alphabeta_search(game, state, cutoff=cutoff_depth(2), euristica=None):
    """Search game to determine best action; use alpha-beta pruning.
    As in [Figure 5.7], this version searches all the way to the leaves."""

    player = state.to_move

    @cache1
    def max_value(state, alpha, beta, depth):
        if game.is_terminal(state):
            return game.utility(state, player), None
        if cutoff(game, state, depth):
            return euristica(game, state, player), None
        v, move = -infinity, None
        for a in game.actions(state):
            v2, _ = min_value(game.result(state, a), alpha, beta, depth+1)
            if v2 > v:
                v, move = v2, a
                alpha = max(alpha, v)
            if v >= beta:
                return v, move
        return v, move

    @cache1
    def min_value(state, alpha, beta, depth):
        if game.is_terminal(state):
            return game.utility(state, player), None
        if cutoff(game, state, depth):
            return euristica(game, state, player), None
        v, move = +infinity, None
        for a in game.actions(state):
            v2, _ = max_value(game.result(state, a), alpha, beta, depth + 1)
            if v2 < v:
                v, move = v2, a
                beta = min(beta, v)
            if v <= alpha:
                return v, move
        return v, move

    return max_value(state, -infinity, +infinity, 0)


def h(game, board, player):
    """Valuta la posizione del gioco e restituisce un valore euristico per il giocatore."""
    score = 0
    avversario = "Red" if player == "Blue" else "Blue"

    # Esempio di valutazione del controllo del territorio
    player_territory = board.count(player)
    opponent_territory = sum(1 for row in board.board for cell in row if cell != player)

    score += player_territory - opponent_territory  # Controllo del territorio

    for row in range(5):
        for col in range(5):
            # print("\n", board.board[row][col])
            if board.board[row][col] is not None:
                if board.board[row][col][0] == player:
                    score += evaluate_position(row, col, board, player)  # Valuta la posizione dell'unità del giocatore
                elif board.board[row][col][0] != player:
                    score -= evaluate_position(row, col, board, avversario)  # Penalizza la posizione dell'unità dell'avversario
                else:
                    score += evaluate_empty_position(row, col, board, player)
                    score -= evaluate_empty_position(row, col, board, avversario)
    '''
    
    '''

    return score

def evaluate_empty_position(row, col, board, player):
    vicini = neighbors(row, col, board)
    vicini_nemici = []
    score = 0
    for (r, c) in vicini:
        if board.board[r][c] is not None:
            if board.board[r][c][0] != player:
                vicini_nemici.append((r, c))
    if len(vicini) == 1:
        return 0
    if len(vicini) == 2:
        (r1, c1) = vicini[0]
        (r2, c2) = vicini[1]
        score += board.board[r1][c1][1] + board.board[r2][c2][1]
        score += 2 if score == 6 else 0
    if len(vicini) == 3:
        (r1, c1) = vicini[0]
        (r2, c2) = vicini[1]
        (r3, c3) = vicini[2]

        score1 = board.board[r1][c1][1] + board.board[r2][c2][1]
        score1 = score1 if score1 <= 6 else 0
        score1 += 2 if score1 == 6 else 0

        score2 = board.board[r2][c2][1] + board.board[r3][c3][1]
        score2 = score2 if score2 <= 6 else 0
        score2 += 2 if score2 == 6 else 0

        score3 = board.board[r1][c1][1] + board.board[r3][c3][1]
        score3 = score3 if score3 <= 6 else 0
        score3 += 2 if score3 == 6 else 0

        score4 = board.board[r1][c1][1] + board.board[r1][c2][1] + board.board[r3][c3][1]
        score4 = score4 if score4 <= 6 else 0
        score4 += 2 if score4 == 6 else 0

        score += max(score1, score2, score3, score4)

    if len(vicini) == 4:
        scores = []
        for (r1, c1), (r2, c2) in vicini:
            if (r1, c1) != (r2, c2):
                s = board.board[r1][c1][1] + board.board[r2][c2][1]
                s = s if s <= 6 else 0
                s += 2 if s == 6 else 0
                scores.append(s)
        score += max(scores)
        scores = []
        for (r1, c1), (r2, c2), (r3, c3) in vicini:
            if (r1, c1) != (r2, c2) and (r1, c1) != (r3, c3) and (r2, c2) != (r3, c3):
                s = board.board[r1][c1][1] + board.board[r2][c2][1] + board.board[r3][c3][1]
                s = s if s <= 6 else 0
                s += 2 if score == 6 else 0
                scores.append(s)
        score += max(scores)

        (r1, c1) = vicini[0]
        (r2, c2) = vicini[1]
        (r3, c3) = vicini[2]
        (r4, c4) = vicini[3]

        s = board.board[r1][c1][1] + board.board[r2][c2][1] + board.board[r3][c3][1] + board.board[r4][c4]
        score += s if s <= 6 else 0
        score += 2 if s == 6 else 0

    return score + len(vicini_nemici)

def evaluate_position(row, col, board, player):
    """Valuta la posizione di un'unità in base alla sua posizione sulla griglia."""
    # Implementa la logica per valutare la posizione
    score = 0
    vicini = neighbors(row, col, board)
    n_vicini = len(vicini)
    if (row, col) in [(0, 0), (0, 4), (4, 0), (4, 4)]:
        score += 2  # Punteggio alto per gli angoli
        if n_vicini == 2:
            score += 2
    elif (row, col) in [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 4), (2, 4), (3, 4), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)]:
        score += 1  # Punteggio per le posizioni ai bordi
        if n_vicini == 3:
            score += 3
    if n_vicini == 4:
        score += 4
    if board.board[row][col][1] == 6:
        score += 3

    for (r, c) in vicini:
        if board.board[r][c][1] == 6:
            score += 1

    return score

def neighbors(row, col, board):
    ret = []
    if col - 1 >= 0:
        ret.append((row, col - 1)) if board.board[row][col -1] is not None else None
    if row + 1 < 5:
        ret.append((row + 1, col)) if board.board[row + 1][col] is not None else None
    if row - 1 >= 0:
        ret.append((row - 1, col)) if board.board[row - 1][col] is not None else None
    if col + 1 < 5:
        ret.append((row, col + 1)) if board.board[row][col + 1] is not None else None
    return ret

def h1(game, state, player):
    """Euristica avanzata combinata per Cephalopod."""
    opponent = "Red" if player == "Blue" else "Blue"
    player_cells = 0
    opponent_cells = 0
    player_pips = 0
    opponent_pips = 0
    player_threat = 0
    opponent_threat = 0
    center_control = 0
    mobility = len(game.actions(state))

    center = state.size // 2

    for r in range(state.size):
        for c in range(state.size):
            cell = state.board[r][c]
            if cell is not None:
                owner, pip = cell

                dist_to_center = abs(r - center) + abs(c - center)
                center_bonus = max(0, 3 - dist_to_center)

                if owner == player:
                    player_cells += 1
                    player_pips += pip
                    center_control += center_bonus

                    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < state.size and 0 <= nc < state.size:
                            neighbor = state.board[nr][nc]
                            if neighbor and neighbor[0] == opponent and neighbor[1] <= 5:
                                opponent_threat += 1
                elif owner == opponent:
                    opponent_cells += 1
                    opponent_pips += pip

                    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < state.size and 0 <= nc < state.size:
                            neighbor = state.board[nr][nc]
                            if neighbor and neighbor[0] == player and neighbor[1] <= 5:
                                player_threat += 1

    score = (
        4 * (player_cells - opponent_cells) +
        1.5 * (player_pips - opponent_pips) +
        0.5 * center_control +
        0.2 * mobility -
        1.0 * player_threat +
        1.0 * opponent_threat
    )
    return score


# ______________________________________________________________________________
# Monte Carlo Tree Search


def monte_carlo_tree_search(state, game, N=1000):
    def select(n):
        """select a leaf node in the tree"""
        if n.children:
            return select(max(n.children.keys(), key=ucb))
        else:
            return n

    def expand(n):
        """expand the leaf node by adding all its children states"""
        if not n.children and not game.is_terminal(n.state):
            n.children = {MCT_Node(state=game.result(n.state, action), parent=n): action
                          for action in game.actions(n.state)}
        return select(n)

    def simulate(game, state):
        """simulate the utility of current state by random picking a step"""
        player = state.to_move
        while not game.is_terminal(state):
            action = random.choice(list(game.actions(state)))
            state = game.result(state, action)
        v = game.utility(state, player)
        return -v

    def backprop(n, utility):
        """passing the utility back to all parent nodes"""
        if utility > 0:
            n.U += utility
        # if utility == 0:
        #     n.U += 0.5
        n.N += 1
        if n.parent:
            backprop(n.parent, -utility)

    root = MCT_Node(state=state)

    for _ in range(N):
        leaf = select(root)
        child = expand(leaf)
        result = simulate(game, child.state)
        backprop(child, result)

    max_state = max(root.children, key=lambda p: p.N)

    return root.children.get(max_state)



