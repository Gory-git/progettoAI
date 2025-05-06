from progetto.codice_gioco.playingStrategies import monte_carlo_tree_search


def playerStrategy(game, state):
    return monte_carlo_tree_search(state, game, 2000)
