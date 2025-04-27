import playingStrategies
#import random
#import game

# The moves of player have the form (x,y), where y is the column number and x the row number (starting with 0)

def playerStrategy (game,state):
    cutOff = 3 # The depth of the search tree. It can be changed to test the performance of the player.

    player = state.to_move
    avversario = "Red" if player == "Blue" else "Blue"
    occupate_player = state.count(player)
    occupate_avversario = state.count(avversario)
    occupate = occupate_player + occupate_avversario

    if occupate >= 6:
        cutOff = 4
    if occupate >= 16:
        cutOff = 5
    if occupate >= 20:
        cutOff = 6
    if occupate >= 22:
        cutOff = 7
    if occupate >= 23:
        cutOff = 8

    print(f"\n cutoff: {cutOff}, occupate: {occupate}")

    # The player uses the alphabeta search algorithm to find the best move.
    value,move = playingStrategies.h_alphabeta_search(game,state,playingStrategies.cutoff_depth(cutOff))
  
    return move

