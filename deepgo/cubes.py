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
"""

# name -> (function, dtype)
reg_cube = {}
reg_label = {}
def register(where, name):
    def registrator(func):
        where[name] = func
        return func
    return registrator

@register(reg_label, 'simple_label')
def get_label_simple(future_moves, boardsize=19):
    player, (row, col) = future_moves.next()
    return np.array((boardsize * row + col,), dtype='uint8')

@register(reg_label, 'expanded_label')
def get_label_exp(future_moves, boardsize=19):
    player, (row, col) = future_moves.next()
    ret = np.zeros((boardsize, boardsize), dtype='uint8')
    ret[row][col] = 1
    return ret

@register(reg_label, 'expanded_label_packed')
def get_label_exp_packed(future_moves, boardsize=19):
    player, (row, col) = future_moves.next()
    label = get_label_exp(move, boardsize)
    return np.packbits(label)

@register(reg_label, '3_moves_lookahead_expanded_label')
def get_label_future3_exp(future_moves, boardsize=19):
    """
    Planes used in
    Yuandong Tian, Yan Zhu, 2015
    Better Computer Go Player with Neural Network and Long-term Prediction
    (arXiv:1511.06410)

    Predicting next 3 moves instead of just one.
    """
    ret = np.zeros((3, boardsize, boardsize), dtype='uint8')

    last_player = None
    for plane, (player, move) in enumerate(islice(future_moves, 3)):
        assert player != last_player
        # pass otw
        if move:
            row, col = move
            ret[plane][row][col] = 1
        last_player = player
    return ret

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

