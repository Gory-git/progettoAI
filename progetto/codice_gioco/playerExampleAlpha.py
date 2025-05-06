import playingStrategies
from progetto.codice_gioco.playingStrategies import monte_carlo_tree_search

#import random
#import game

# The moves of player have the form (x,y), where y is the column number and x the row number (starting with 0)

def playerStrategy (game,state):

    player = state.to_move
    avversario = "Red" if player == "Blue" else "Blue"
    occupate_player = state.count(player)
    occupate_avversario = state.count(avversario)
    occupate = occupate_player + occupate_avversario

    if occupate <= 18:
        cutOff = 4
        value, move = playingStrategies.h_alphabeta_search(game, state, playingStrategies.cutoff_depth(cutOff),playingStrategies.h)
        return move
    if occupate <= 20:
        cutOff = 5
        value,move = playingStrategies.h_alphabeta_search(game,state,playingStrategies.cutoff_depth(cutOff), playingStrategies.h)
        return move
    if occupate <= 21:
        cutOff = 6
        value,move = playingStrategies.h_alphabeta_search(game,state,playingStrategies.cutoff_depth(cutOff), playingStrategies.h)
        return move
    if occupate <= 22:
        cutOff = 7
        value, move = playingStrategies.h_alphabeta_search(game, state, playingStrategies.cutoff_depth(cutOff),playingStrategies.h)
        return move
    if occupate <= 23:
        cutOff = 8
        value, move = playingStrategies.h_alphabeta_search(game, state, playingStrategies.cutoff_depth(cutOff),playingStrategies.h)
        return move
    if occupate <= 24:
        cutOff = 9
        value, move = playingStrategies.h_alphabeta_search(game, state, playingStrategies.cutoff_depth(cutOff),playingStrategies.h)
        return move
    if occupate <= 25:
        cutOff = 10
        value, move = playingStrategies.h_alphabeta_search(game, state, playingStrategies.cutoff_depth(cutOff),playingStrategies.h)
        return move


        #cutOff = 6
        #value,move = playingStrategies.h_alphabeta_search(game,state,playingStrategies.cutoff_depth(cutOff), playingStrategies.h)
        #return move
    # if occupate <= 22:
    #     return monte_carlo_tree_search(state, game, 4000)
    # if occupate <= 25:
    #     return monte_carlo_tree_search(state, game, 10000)
    # return None

    # print(f"\n cutoff: {cutOff}, occupate: {occupate}")

    # The player uses the alphabeta search algorithm to find the best move.


