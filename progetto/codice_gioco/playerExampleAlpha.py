import math

infinity = math.inf

def playerStrategy (game,state):

    player = state.to_move
    avversario = "Red" if player == "Blue" else "Blue"
    occupate_player = state.count(player)
    occupate_avversario = state.count(avversario)
    occupate = occupate_player + occupate_avversario
    if occupate <= 8:
        cutOff = 3
        value, move = h_alphabeta_search(game, state, cutoff_depth(cutOff))
        return move
    if occupate <= 18:
        cutOff = 4
        value, move = h_alphabeta_search(game, state, cutoff_depth(cutOff))
        return move
    if occupate <= 21:
        cutOff = 5
        value,move = h_alphabeta_search(game,state,cutoff_depth(cutOff))
        return move
    if occupate <= 22:
        cutOff = 7
        value, move = h_alphabeta_search(game, state, cutoff_depth(cutOff))
        return move
    if occupate <= 23:
        cutOff = 8
        value, move = h_alphabeta_search(game, state, cutoff_depth(cutOff))
        return move
    if occupate <= 24:
        cutOff = 9
        value, move = h_alphabeta_search(game, state, cutoff_depth(cutOff))
        return move
    if occupate <= 25:
        cutOff = 10
        value, move = h_alphabeta_search(game, state, cutoff_depth(cutOff))
        return move
    return None

def cache1(function):
    "Like lru_cache(None), but only considers the first argument of function."
    cache = {}
    def wrapped(x, *args):
        if x not in cache:
            cache[x] = function(x, *args)
        return cache[x]
    return wrapped

def cutoff_depth(d):
    """A cutoff function that searches to depth d."""
    return lambda game, state, depth: depth > d

def h_alphabeta_search(game, state, cutoff=cutoff_depth(2)):
    """Search game to determine best action; use alpha-beta pruning.
    As in [Figure 5.7], this version searches all the way to the leaves."""

    player = state.to_move

    @cache1
    def max_value(state, alpha, beta, depth):
        if game.is_terminal(state):
            return game.utility(state, player), None
        if cutoff(game, state, depth):
            return h(game, state, player), None
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
            return h(game, state, player), None
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
    opponent_territory = board.count(avversario)

    score += 3.5 * (player_territory - opponent_territory) # Controllo del territorio
    somma_p = 0
    somma_a = 0
    for row in range(5):
        for col in range(5):
            # print("\n", board.board[row][col])
            if board.board[row][col] is not None:
                val = board.board[row][col][1]
                occ = board.board[row][col][0]
                if occ == player:
                    score += evaluate_position(row, col, board)  # Valuta la posizione dell'unità del giocatore
                    somma_p += val if 1 < val < 6 else 0
                if occ == avversario:
                    score -= evaluate_position_enemy(row, col, board) # Penalizza la posizione dell'unità dell'avversario
                    somma_a += val if 1 < val < 6 else 0

    return score + 1.5 * (somma_p - somma_a)

def evaluate_position_enemy(row, col, board):
    """Valuta la posizione di un'unità in base alla sua posizione sulla griglia."""
    # Implementa la logica per valutare la posizione


    centro = 0
    medio = 0
    angolo = 0
    bordo = 0
    sei = 0
    vicini_occupati = 0
    minacciato = 0


    vicini = neighbors(row, col, board)
    n_vicini = len(vicini)
    if row == col == 2: # and player_territory + opponent_territory == 1: # Punteggio molto alto per il centro
        centro += 1
    if (row, col) in [(0, 0), (0, 4), (4, 0), (4, 4)]:
        angolo += 1# Punteggio alto per gli angoli
        if n_vicini == 2:
            vicini_occupati += 1
    elif (row, col) in [(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 4), (1, 0), (2, 0), (3, 0), (4, 1), (4, 2), (4, 3)]:
        bordo += 1  # Punteggio per le posizioni ai bordi
        if n_vicini == 3:
            vicini_occupati += 1
    elif (row, col) in [(1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2), (3, 3)]:
        medio += 1
    if n_vicini == 4:
        vicini_occupati += 1
    if board.board[row][col][1] == 6:
        sei += 1.5

    for (r, c) in vicini:
        if board.board[r][c] is None:
            vicini_del_vicino = neighbors(r, c, board)
            for (rv, cv) in vicini_del_vicino:
                if r != rv and c != cv:
                    if board.board[row][col][1] + board.board[rv][cv][1] <= 6:
                        minacciato += 1
                    if board.board[row][col][1] + board.board[rv][cv][1] == 6:
                        minacciato += 0.5

    return (
        2 * centro +
        2.5 * medio +
        3 * angolo +
        1.5 * bordo +
        5 * sei -
        3 * minacciato +
        1.25 * vicini_occupati
    )

def evaluate_position(row, col, board):
    """Valuta la posizione di un'unità in base alla sua posizione sulla griglia."""
    # Implementa la logica per valutare la posizione


    centro = 0
    medio = 0
    angolo = 0
    bordo = 0
    sei = 0
    vicini_occupati = 0
    minacciato = 0


    vicini = neighbors(row, col, board)
    n_vicini = len(vicini)
    if row == col == 2: # and player_territory + opponent_territory == 1: # Punteggio molto alto per il centro
        centro += 1
    if (row, col) in [(0, 0), (0, 4), (4, 0), (4, 4)]:
        angolo += 1# Punteggio alto per gli angoli
        if n_vicini == 2:
            vicini_occupati += 1
    elif (row, col) in [(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 4), (1, 0), (2, 0), (3, 0), (4, 1), (4, 2), (4, 3)]:
        bordo += 1  # Punteggio per le posizioni ai bordi
        if n_vicini == 3:
            vicini_occupati += 1
    elif (row, col) in [(1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2), (3, 3)]:
        medio += 1
    if n_vicini == 4:
        vicini_occupati += 1
    if board.board[row][col][1] == 6:
        sei += 1.5

    for (r, c) in vicini:
        if board.board[r][c] is None:
            vicini_del_vicino = neighbors(r, c, board)
            for (rv, cv) in vicini_del_vicino:
                if r != rv and c != cv:
                    if board.board[row][col][1] + board.board[rv][cv][1] <= 6:
                        minacciato += 1
                    if board.board[row][col][1] + board.board[rv][cv][1] == 6:
                        minacciato += 0.5

    return (
        2 * centro +
        2.5 * medio +
        3 * angolo +
        1.5 * bordo +
        4.5 * sei -
        3 * minacciato +
        1.25 * vicini_occupati
    )

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