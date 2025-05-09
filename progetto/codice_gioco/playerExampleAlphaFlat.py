import itertools
import playingStrategies


# Gruppo MAD
# Nome: Marco Pio Agatino D'Agosta 268999, Anastasia Martucci 271316, Domenico Macrì 269798


def h(game, state, player):
    board = state.board
    n = state.size
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    enemy = "Red" if player == "Blue" else "Blue"

    center = n // 2

    W_DADI = 1.5
    W_VAL_DADI = 0.3
    W_POTENZIALE_CATTURA = 1.8
    W_CATTURA = 2.0
    W_LIBERTA = 0.2
    W_CENTRO = 0.4
    W_ANGOLI = 0.1
    W_SEI = 0.5
    W_VULNERABILITA = 1.0
    W_VICINI = 0.3

    alleato_count = opp_count = alleato_valoreDadi = opp_valoreDadi = 0
    sei_alleati = sei_opp = 0
    spazio_alleato = spazio_opp = 0
    angoli_alleato = angoli_opp = 0
    potenziale_cattura_alleato = potenziale_cattura_opp = 0
    cattura_reale_alleato = cattura_reale_opp = 0
    controllo_centro = 0
    vulnerabilita = 0
    bonus_vicini = 0

    for r in range(n):
        for c in range(n):
            cell = board[r][c]
            if cell is None:
                continue
            owner, valoreDadi = cell
            is_alleato = owner == player
            if is_alleato:
                alleato_count += 1
                alleato_valoreDadi += valoreDadi
                if valoreDadi == 6:
                    sei_alleati += 1
            else:
                opp_count += 1
                opp_valoreDadi += valoreDadi
                if valoreDadi == 6:
                    sei_opp += 1

            dist_centro = abs(r - center) + abs(c - center)
            if is_alleato:
                controllo_centro += (n - dist_centro)
            else:
                controllo_centro -= (n - dist_centro)

            alleati = 0
            liberta = 0
            minacce = []
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n:
                    vicini = board[nr][nc]
                    if vicini is None:
                        liberta += 1
                    else:
                        n_owner, n_valoreDadi = vicini
                        if n_owner == owner:
                            alleati += 1
                        elif n_owner != owner:
                            minacce.append(n_valoreDadi)
            if is_alleato:
                spazio_alleato += liberta
                bonus_vicini += alleati
                if sum(minacce) >= valoreDadi:
                    vulnerabilita -= 1
            else:
                spazio_opp += liberta
                bonus_vicini -= alleati
                if sum(minacce) >= valoreDadi:
                    vulnerabilita += 1

            if (r, c) in [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]:
                if is_alleato:
                    angoli_alleato += 1
                else:
                    angoli_opp += 1

    # Catture potenziali
    for r in range(n):
        for c in range(n):
            if board[r][c] is not None:
                continue
            vicini = []
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n:
                    nb = board[nr][nc]
                    if nb is not None:
                        vicini.append(((nr, nc), nb))
            if len(vicini) >= 2:
                for k in range(2, len(vicini) + 1):
                    for subset in itertools.combinations(vicini, k):
                        somma_valoreDadi = sum(p[1] for _, p in subset)
                        if somma_valoreDadi <= 6:
                            has_enemy = any(p[0] == enemy for _, p in subset)
                            has_alleato = any(p[0] == player for _, p in subset)
                            if has_enemy:
                                potenziale_cattura_alleato += 1
                                if all(p[0] == enemy for _, p in subset):
                                    cattura_reale_alleato += 1
                            if has_alleato:
                                potenziale_cattura_opp += 1
                                if all(p[0] == player for _, p in subset):
                                    cattura_reale_opp += 1

    score = 0
    score += W_DADI * (alleato_count - opp_count)
    score += W_VAL_DADI * (alleato_valoreDadi - opp_valoreDadi)
    score += W_POTENZIALE_CATTURA * (potenziale_cattura_alleato - potenziale_cattura_opp)
    score += W_CATTURA * (cattura_reale_alleato - cattura_reale_opp)
    score += W_LIBERTA * (spazio_alleato - spazio_opp)
    score += W_CENTRO * controllo_centro
    score += W_ANGOLI * (angoli_alleato - angoli_opp)
    score += W_SEI * (sei_alleati - sei_opp)
    score += W_VULNERABILITA * vulnerabilita
    score += W_VICINI * bonus_vicini

    return score


playingStrategies.h = h


def playerStrategy(game, state):
    cutOff = 3
    value, move = playingStrategies.h_alphabeta_search(game, state, playingStrategies.cutoff_depth(cutOff))

    return move

