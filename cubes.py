import logging
import array
import numpy as np

import gomill
import go_strings

def get_label_simple(move, boardsize=19):
    row, col = move
    return np.array((boardsize * row + col,), dtype='uint8')

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

def get_cube_simple(board, ko_point, player):
    """
    The same information as in get_cube_clark_storkey_2014()
    """
    colors, strings, liberties = go_strings.board2strings(board)
    opponent = gomill.common.opponent_of(player)

    def is_our_stone((row, col)):
        return board.get(row, col) == player
    def is_enemy_stone((row, col)):
        return board.get(row, col) == opponent
    def is_empty((row, col)):
        return board.get(row, col) == None

    plane_functions = [ (lambda pt : is_our_stone(pt)), 
                        (lambda pt : is_enemy_stone(pt)), 
                        (lambda pt : is_empty(pt)), 
                        (lambda pt : not is_empty(pt) and len(liberties[strings[pt]]) == 1),
                        (lambda pt : not is_empty(pt) and len(liberties[strings[pt]]) == 2),
                        (lambda pt : not is_empty(pt) and len(liberties[strings[pt]]) >= 3),
                        (lambda pt : pt == ko_point) ]

    cube = np.zeros((len(plane_functions), board.side, board.side), dtype='uint8')
    
    for plane, planefc in enumerate(plane_functions):
        for row in xrange(board.side):
            for col in xrange(board.side):
                cube[plane][row][col] = planefc(pt)
            
    return cube

def get_cube_more_lib(board, ko_point, player):
    """
    """
    colors, strings, liberties = go_strings.board2strings(board)
    opponent = gomill.common.opponent_of(player)

    def is_our_stone((row, col)):
        return board.get(row, col) == player
    def is_enemy_stone((row, col)):
        return board.get(row, col) == opponent
    def is_empty((row, col)):
        return board.get(row, col) == None

    plane_functions = [ (lambda pt : is_our_stone(pt)), 
                        (lambda pt : is_enemy_stone(pt)), 
                        # atari
                        (lambda pt : not is_empty(pt) and len(liberties[strings[pt]]) == 1),
                        # num of liberties
                        (lambda pt : not is_empty(pt) and len(liberties[strings[pt]])),
                        (lambda pt : pt == ko_point) ]

    cube = np.zeros((len(plane_functions), board.side, board.side), dtype='uint8')
    
    for plane, planefc in enumerate(plane_functions):
        for row in xrange(board.side):
            for col in xrange(board.side):
                cube[plane][row][col] = planefc(pt)
            
    return cube
