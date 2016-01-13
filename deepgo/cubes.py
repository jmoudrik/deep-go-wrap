import logging
import array
import numpy as np
from itertools import islice

import gomill
import analyze_board
from static_planes import get_border_mark, get_sqd_from_center
from utils import raw_history
from rank import Rank, BrWr

"""
    Core cube processing

    Returned cubes are usually in unpacked float form,
    unsuitable to be directly stored in a file, because it gets rather big.

    There are two families
        * labels
        * cubes

    1) labels
        * the y's for prediction
        * args:
            future    iterator of future moves...
            board
            player          'w' or 'b' -- next move's player
    2) cubes
        * the X's for prediction
        * args:
            state           game state deepgo.states.State
                            (see deepgo/state/ for details)
            player          'w' or 'b'
"""

class XX:
    def __init__(self, n=0):
        self.n=0
    def __call__(self):
        old = self.n
        self.n += 1
        return old

# name -> (function, dtype)
reg_cube = {}
reg_label = {}
def register(where, name):
    def registrator(func):
        where[name] = func
        return func
    return registrator

#
# Labels
#

@register(reg_label, 'simple_label')
def get_label_simple(s, player):
    assert s.future
    player_next, (row, col) = s.future[0]
    assert player == player_next
    return np.array((s.board.side * row + col,), dtype='uint8')

@register(reg_label, 'expanded_label')
def get_label_exp(s, player):
    assert s.future
    player_next, (row, col) = s.future[0]
    assert player == player_next
    ret = np.zeros((s.board.side, s.board.side), dtype='uint8')
    ret[row][col] = 1
    return ret

@register(reg_label, 'expanded_label_packed')
def get_label_exp_packed(s, player):
    assert s.future
    player_next, (row, col) = s.future[0]
    assert player == player_next
    label = get_label_exp((row, col), s.board.side)
    return np.packbits(label)

@register(reg_label, '3_moves_lookahead_expanded_label')
def get_label_future3_exp(s, player):
    """
    Planes used in
    Yuandong Tian, Yan Zhu, 2015
    Better Computer Go Player with Neural Network and Long-term Prediction
    (arXiv:1511.06410)

    Predicting next 3 moves instead of just one.
    """
    ret = np.zeros((3, s.board.side, s.board.side), dtype='uint8')

    last_player = gomill.common.opponent_of(player)
    for plane, (player_next, move) in enumerate(s.future[:3]):
        assert player_next != last_player
        # pass otw
        if move:
            row, col = move
            ret[plane][row][col] = 1
        last_player = player_next
    return ret

@register(reg_label, 'correct_moves')
def get_label_correct(s, player):
    return analyze_board.board2correct_move_mask(s.board, player)

#
# Cubes
#

@register(reg_cube, 'nop')
def get_cube_nop(state, player):
    return np.zeros((1, state.board.side, state.board.side), dtype='float32')

@register(reg_cube, 'clark_storkey_2014')
def get_cube_clark_storkey_2014(*args):
    """
    Planes compatible with the Clark and Storkey 2014 paper
    (arXiv:1412.3409v1)
    """

    return get_cube_basic_7_channel(*args)

@register(reg_cube, 'basic_7_channel')
def get_cube_basic_7_channel(state, player):
    cube = np.zeros((7, state.board.side, state.board.side), dtype='uint8')

    # count liberties
    string_lib = analyze_board.board2string_lib(state.board)
    lib_count = analyze_board.liberties_count(state.board, string_lib)

    # mask for different colors
    empty, friend, enemy = analyze_board.board2color_mask(state.board, player)

    our_liberties = friend * lib_count
    enemy_liberties = enemy * lib_count

    cube[0] = our_liberties == 1
    cube[1] = our_liberties == 2
    cube[2] = our_liberties >= 3
    cube[3] = enemy_liberties == 1
    cube[4] = enemy_liberties == 2
    cube[5] = enemy_liberties >= 3
    if state.ko_point is not None:
        ko_row, ko_col = state.ko_point
        cube[6][ko_row][ko_col] = 1

    return cube

@register(reg_cube, 'clark_storkey_2014_packed')
def get_cube_clark_storkey_2014_packed(*args):
    cube = get_cube_clark_storkey_2014(*args)
    return np.packbits(cube)

@register(reg_cube, 'deepcl')
def get_cube_deepcl(*args):
    """v2 version compatible planes
    https://github.com/hughperkins/kgsgo-dataset-preprocessor
    """

    cube = get_cube_basic_7_channel(*args)

    return np.array(255 * cube, dtype='float32')

@register(reg_cube, 'tian_zhu_2015')
def get_cube_tian_zhu_2015(state, player):
    """
    Planes compatible with the
    Yuandong Tian, Yan Zhu, 2015
    Better Computer Go Player with Neural Network and Long-term Prediction
    (arXiv:1511.06410)
    """
    cube = np.zeros((25, state.board.side, state.board.side), dtype='float32')

    # count liberties
    string_lib = analyze_board.board2string_lib(state.board)
    lib_count = analyze_board.liberties_count(state.board, string_lib)

    # mask for different colors
    empty, friend, enemy = analyze_board.board2color_mask(state.board, player)

    our_liberties = friend * lib_count
    enemy_liberties = enemy * lib_count


    cube[0] = our_liberties == 1
    cube[1] = our_liberties == 2
    cube[2] = our_liberties >= 3
    cube[3] = enemy_liberties == 1
    cube[4] = enemy_liberties == 2
    cube[5] = enemy_liberties >= 3
    if state.ko_point is not None:
        ko_row, ko_col = state.ko_point
        cube[6][ko_row][ko_col] = 1

    cube[7] = friend * 1
    cube[8] = enemy * 1
    cube[9] = empty * 1

    # watch out, history since it gives -1 for empty points
    history = np.exp(- 0.1 * raw_history(state.board, state.history))
    cube[10] = friend * history
    cube[11] = enemy * history

    # rank_planes
    enemy_rank = state.ranks.wr if player == 'b' else state.ranks.br
    # key maps: (30k, 1k) = (30, 1), (1d, 9d) = (0, -8), (1p, 9p) = (-9, ..)
    # in case of None rank, make all rank planes 0
    enemy_rank_key = enemy_rank.key() if enemy_rank is not None else 1
    for r in xrange(9):
        # one plane per dan, pros have all ones
        if enemy_rank_key == -r or enemy_rank_key < -8:
            cube[12 + r][:] = 1

    cube[21] = get_border_mark(state.board.side)
    cube[22] = np.exp(-0.5 * get_sqd_from_center(state.board.side))

    # distances from stones ~ cfg
    dist_friend, dist_enemy = analyze_board.board2dist_from_stones(state.board, player)
    cube[23] = empty * (dist_friend < dist_enemy)
    cube[24] = empty * (dist_friend > dist_enemy)

    return cube

@register(reg_cube, 'detlef')
def get_cube_detlef(state, player):
    """
    Planes compatible with the
    CNN kindly provided by Detlef Schmicker. See
    http://computer-go.org/pipermail/computer-go/2015-December/008324.html

    The net should (as of January 2016) be available here:
    http://physik.de/CNNlast.tar.gz

    Description:
    1
    2
    3
    > 4 libs playing color
    1
    2
    3
    > 4 libs opponent color
    Empty points
    last move
    second last move
    third last move
    forth last move
    """
    cube = np.zeros((13, state.board.side, state.board.side), dtype='float32')

    # count liberties
    string_lib = analyze_board.board2string_lib(state.board)
    lib_count = analyze_board.liberties_count(state.board, string_lib)

    # mask for different colors
    empty, friend, enemy = analyze_board.board2color_mask(state.board, player)

    our_liberties = friend * lib_count
    enemy_liberties = enemy * lib_count

    cube[0] = our_liberties == 1
    cube[1] = our_liberties == 2
    cube[2] = our_liberties == 3
    cube[3] = our_liberties >= 4
    cube[4] = enemy_liberties == 1
    cube[5] = enemy_liberties == 2
    cube[6] = enemy_liberties == 3
    cube[7] = enemy_liberties >= 4

    cube[8] = empty * 1

    # watch out, history gives -1 for empty points
    history = raw_history(state.board, state.history)
    cube[9]  = 1*(history == 1)
    cube[10] = 1*(history == 2)
    cube[11] = 1*(history == 3)
    cube[12] = 1*(history == 4)

    return cube

@register(reg_cube, 'detlefko')
def get_cube_detlefko(state, player):
    cube = np.zeros((14, state.board.side, state.board.side), dtype='float32')

    string_lib = analyze_board.board2string_lib(state.board)
    lib_count = analyze_board.liberties_count(state.board, string_lib)

    empty, friend, enemy = analyze_board.board2color_mask(state.board, player)

    our_liberties, enemy_liberties = friend * lib_count, enemy * lib_count

    cube[0] = our_liberties == 1
    cube[1] = our_liberties == 2
    cube[2] = our_liberties == 3
    cube[3] = our_liberties >= 4
    cube[4] = enemy_liberties == 1
    cube[5] = enemy_liberties == 2
    cube[6] = enemy_liberties == 3
    cube[7] = enemy_liberties >= 4

    cube[8] = empty * 1

    # watch out, history gives -1 for empty points
    history = raw_history(state.board, state.history)
    cube[9]  = 1*(history == 1)
    cube[10] = 1*(history == 2)
    cube[11] = 1*(history == 3)
    cube[12] = 1*(history == 4)
    if state.ko_point is not None:
        ko_row, ko_col = state.ko_point
        cube[13][ko_row][ko_col] = 1

    return cube


@register(reg_cube, 'detlefko_conthist')
def get_cube_detlefko_conthist(state, player):
    cube = np.zeros((12, state.board.side, state.board.side), dtype='float32')

    string_lib = analyze_board.board2string_lib(state.board)
    lib_count = analyze_board.liberties_count(state.board, string_lib)

    empty, friend, enemy = analyze_board.board2color_mask(state.board, player)

    our_liberties, enemy_liberties = friend * lib_count, enemy * lib_count

    cube[0] = our_liberties == 1
    cube[1] = our_liberties == 2
    cube[2] = our_liberties == 3
    cube[3] = our_liberties >= 4
    cube[4] = enemy_liberties == 1
    cube[5] = enemy_liberties == 2
    cube[6] = enemy_liberties == 3
    cube[7] = enemy_liberties >= 4

    cube[8] = empty * 1

    if state.ko_point is not None:
        ko_row, ko_col = state.ko_point
        cube[9][ko_row][ko_col] = 1

    # watch out, history since it gives -1 for empty points
    history = np.exp(- 0.1 * raw_history(state.board, state.history))
    cube[10] = friend * history
    cube[11] = enemy * history
    return cube

if __name__ == "__main__":
    def test_cube():
        import gomill.boards,  gomill.ascii_boards, gomill.common

        from state import State

        logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                        level=logging.DEBUG)

        b = gomill.boards.Board(3)
        moves = [('b', (0,1)),
                 ('w', (2,0)),
                 ('b', (1,1))]

        b2 = gomill.boards.Board(3)
        for col, (x,y) in moves:
            b.play(x,y, col)
            b2.play(2-x,y, col) # mirror s.t. we have the same coords
                                # with numpy arrays

        logging.debug("\n"+gomill.ascii_boards.render_board(b2))
        cube = get_cube_tian_zhu_2015(State(b, None, moves, BrWr(Rank.from_string('1p'), None)), 'w')
        for a in xrange(cube.shape[0]):
            logging.debug("%d\n%s"%(a,cube[a]))
        logging.debug("\n"+ str(get_label_future3_exp(moves, 3)))

    def time_cube():
        logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                            level=logging.DEBUG)
        import time
        from state import State
        with open("../test_sgf/test1.sgf", 'r') as fin:
            game = gomill.sgf.Sgf_game.from_string(fin.read())

        tt0 = 0
        tt1 = 0
        it = 0

        for i in xrange(2):
            board, movepairs = gomill.sgf_moves.get_setup_and_moves(game)
            history = []
            for color, move in movepairs:
                if move:
                    it += 1
                    row, col = move
                    board.play(row, col, color)


                    s = time.clock()
                    c1 = get_cube_tian_zhu_2015(State(board, None, history, BrWr(Rank.from_string('1p'), None)), gomill.common.opponent_of(color))
                    #c1 = get_cube_clark_storkey_2014(State(board, None, history), gomill.common.opponent_of(color))
                    tt0 += time.clock() - s

                    #s = time.clock()
                    #c2 = get_cube_clark_storkey_2014_2(State(board, None, history), gomill.common.opponent_of(color))
                    #tt1 += time.clock() - s

                    #assert np.array_equal(c1, c2)

                    history.append((color, move))

        logging.debug("tt0 = %.3f, %.5f per one "%(tt0, tt0/it))
        logging.debug("tt1 = %.3f, %.5f per one "%(tt1, tt1/it))

    import cProfile
    cProfile.run("time_cube()")

