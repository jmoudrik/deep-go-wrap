import logging
import array
import numpy as np

import gomill
import go_strings

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
    
    The returned value is pretty inefficient, since only 1 bit out of each
    uint8 is used; so np.packbits should be probably applied before storing
    the data to disk.
    """
    colors, strings, liberties = go_strings.board2strings(board)
    opponent = gomill.common.opponent_of(player)

    def is_our_stone((row, col)):
        return board.get(row, col) == player
    def is_enemy_stone((row, col)):
        return board.get(row, col) == opponent

    plane_functions = [ (lambda pt : (is_our_stone(pt) and len(liberties[strings[pt]]) == 1)),
                        (lambda pt : (is_our_stone(pt) and len(liberties[strings[pt]]) == 2)),
                        (lambda pt : (is_our_stone(pt) and len(liberties[strings[pt]]) >= 3)),
                        (lambda pt : (is_enemy_stone(pt) and len(liberties[strings[pt]]) == 1)),
                        (lambda pt : (is_enemy_stone(pt) and len(liberties[strings[pt]]) == 2)),
                        (lambda pt : (is_enemy_stone(pt) and len(liberties[strings[pt]]) >= 3)),
                        (lambda pt : (pt == ko_point)) ]

    cube = np.zeros((len(plane_functions), board.side, board.side), dtype='uint8')
    
    for plane, planefc in enumerate(plane_functions):
        for row in xrange(board.side):
            for col in xrange(board.side):
                cube[plane][row][col] = planefc((row, col))
            
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
        
    test_cube()
    