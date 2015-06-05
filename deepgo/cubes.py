import logging
import array
import numpy as np

import gomill
import analyze_board

# name -> (function, dtype)
reg_cube = {}
reg_label = {}
def register(where, name):
    def registrator(func):
        where[name] = func
        return func
    return registrator

@register(reg_label, 'simple_label')
def get_label_simple(move, boardsize=19):
    row, col = move
    return np.array((boardsize * row + col,), dtype='uint8')

@register(reg_label, 'expanded_label')
def get_label_exp(move, boardsize=19):
    row, col = move
    ret = np.zeros((boardsize, boardsize), dtype='uint8')
    ret[row][col] = 1
    return ret

@register(reg_label, 'expanded_label_packed')
def get_label_exp_packed(move, boardsize=19):
    label = get_label_exp(move, boardsize)
    return np.packbits(label)

@register(reg_cube, 'clark_storkey_2014')
def get_cube_clark_storkey_2014(board, ko_point, player):
    """
    Planes compatible with the Clark and Storkey 2014 paper
    (arXiv:1412.3409v1)
    
    The returned cube is pretty inefficient (in terms of data storage),
    since only 1 bit out of each uint8 is used; so np.packbits or compression
    should be probably applied before storing the data to disk.
    """
    cube = np.zeros((7, board.side, board.side), dtype='uint8')
    
    # count liberties
    string_lib = analyze_board.board2string_lib(board)
    lib_count = analyze_board.liberties_count(board, string_lib)
    
    # mask for different colors
    empty, friend, enemy = analyze_board.board2color_mask(board, player)
    
    our_liberties = friend * lib_count
    enemy_liberties = enemy * lib_count
    
    cube[0] = our_liberties == 1
    cube[1] = our_liberties == 2
    cube[2] = our_liberties >= 3
    cube[3] = enemy_liberties == 1
    cube[4] = enemy_liberties == 2
    cube[5] = enemy_liberties >= 3
    if ko_point is not None:
        ko_row, ko_col = ko_point
        cube[6][ko_row][ko_col] = 1
    
    return cube

@register(reg_cube, 'clark_storkey_2014_packed')
def get_cube_clark_storkey_2014_packed(*args):
    cube = get_cube_clark_storkey_2014(*args)
    return np.packbits(cube)

@register(reg_cube, 'deepcl')
def get_cube_deepcl(board, ko_point, player):
    """v2 version compatible planes
    https://github.com/hughperkins/kgsgo-dataset-preprocessor
    
    Yet, the returned data is in unpacked float form, not the original binary encoding.
    Therefore, the returned data is not very convenient to be directly stored in
    a file, because it gets rather big. One position has size of
    19*19*7*4 B = 10108 B, so one game of ~200 moves corresponds rouhly to 2MB of data.
    """
    
    cube = get_cube_clark_storkey_2014(board, ko_point, player)
            
    return np.array(255 * cube, dtype='float32')

if __name__ == "__main__":
    def test_cube():
        import gomill.boards,  gomill.ascii_boards, gomill.common
        
        logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                        level=logging.DEBUG)
        
        b = gomill.boards.Board(3)
        b.play(1, 1, "b")
        b.play(0, 1, "b")
        b.play(2, 0, "w")
        
        logging.debug("\n"+gomill.ascii_boards.render_board(b))
        
        logging.debug("\n"+ str(get_cube_deepcl(b, None, 'w')))
        
    def time_cube():
        logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                            level=logging.DEBUG)
        import time
        with open("test_sgf/test1.sgf", 'r') as fin:
            game = gomill.sgf.Sgf_game.from_string(fin.read())
        
        tt0 = 0
        tt1 = 0
        it = 0
        
        for i in xrange(10):
            board, movepairs = gomill.sgf_moves.get_setup_and_moves(game)
            for color, move in movepairs:
                if move:
                    it += 1
                    row, col = move
                    board.play(row, col, color)
        
                    
                    s = time.clock()
                    c1 = get_cube_clark_storkey_2014(board, None, gomill.common.opponent_of(color))
                    tt0 += time.clock() - s
                    
                    #s = time.clock()
                    #c2 = get_cube_clark_storkey_2014_2(board, None, gomill.common.opponent_of(color))
                    #tt1 += time.clock() - s
                    
                    #assert np.array_equal(c1, c2)
                    
        logging.debug("tt0 = %.3f, %.5f per one "%(tt0, tt0/it))
        logging.debug("tt1 = %.3f, %.5f per one "%(tt1, tt1/it))
    time_cube()
    