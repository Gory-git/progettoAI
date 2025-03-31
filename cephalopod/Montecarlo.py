import random

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
            if node.state.current_player == "X":  # Se il nodo è di "X", vogliamo massimizzare il punteggio
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