from collections import namedtuple

from rank import BrWr

# this is the state which is passed to the cubes
State = namedtuple('State', 'board ko_point history future ranks')

def gomill_gamestate2state(game_state):
    return State(game_state.board,
                 game_state.ko_point,
                 game_state.move_history,
                 [],
                 BrWr(None, None))


