from itertools import product
import string
import logging
import numpy as np
from collections import namedtuple

import gomill
from gomill import boards, sgf, sgf_moves

NBCOORD_DIAG = tuple(product((1, -1), (1, -1)))
NBCOORD = ((-1,0), (1,0), (0,1), (0,-1))

# StringLib.string = {}
# coord => string number
# StringLib.liberties = {}
# string number => set of liberties coords
StringLib = namedtuple('StringLib', 'string liberties')

# NbInfo for candidate moves (= coord is a currently empty intersection)
# NbInfo.liberties = {}
# coord => set of liberties coords in the direct neighborhood
# NbInfo.friends = {}
# coord => set of friend strings in the direct nbhood (string = string number)
# NbInfo.enemies = {}
# coord => set of enemy strings in the direct nbhood (string = string number)
NbInfo = namedtuple('NbInfo', 'liberties friends enemies')

def coord_onboard(board, (row, col)):
    return row >= 0 and col >= 0 and row < board.side and col < board.side

def iter_nbhs(board, (row, col)):
    for dx, dy in NBCOORD:
        nbx, nby = row + dx, col + dy
        if coord_onboard(board, (nbx, nby)):
            yield nbx, nby

def board2string_lib(board):
    """
    Divides board into strings and computes sets of their liberties.
    Takes O(N) time and space, where N is the number of occupied points.
    Algorithm is a simple dfs.
    
    :returns: StringLib 
    """
    
    # pt => color
    colors = dict((pt, color) for (color, pt) in board.list_occupied_points() )
    # pt => string number
    visited = {}
    # string number => set of liberties
    liberties = {}
    i = -1
    for pt in colors.keys():
        if pt in visited:
            continue
        # new string
        i += 1
        # dfs
        fringe = [pt]
        while fringe:
            n = fringe.pop()
            if n in visited:
                continue
            visited[n] = i
            # dfs over unvisited nbhood of the same color
            for nb in iter_nbhs(board, n):
                # track liberty
                if not nb in colors:
                    liberties.setdefault(i, set()).add(nb)
                    continue
                if nb in visited:
                    # better be safe
                    assert ((colors[nb] != colors[n])
                            or visited[nb] == visited[n])
                    continue
                if colors[nb] == colors[n]:
                    fringe.append(nb)
                    
    return StringLib(visited, liberties)

def analyze_nbhood(board, player, string_lib):
    """
    Analyses neighborhood of candidate moves (=empty intersections)
    
    :returns: NbInfo
    """
    # coord = (row,col) which points to an empty intersection
    # coord => set of liberties in the direct neighborhood
    nb_libs = {}
    # coord => set of friend strings in the direct nbhood
    nb_friend = {}
    # coord => set of enemy strings in the direct nbhood
    nb_enemy = {}
    
    # fill in correct moves
    for row in xrange(board.side):
        for col in xrange(board.side):
            move = (row, col)
            if board.get(row, col):
                continue
            
            for nb in iter_nbhs(board, (row, col)):
                row_nb, col_nb = nb
                nb_color = board.get(row_nb, col_nb)
                # liberty
                if not nb_color:
                    nb_libs.setdefault(move, set()).add(nb)
                    continue
                    
                # neighbor is not a liberty, so it is a stone, so
                # it belongs to a string
                si = string_lib.string[nb]
                if nb_color == player:
                    # connect to a friend
                    nb_friend.setdefault(move, set()).add(si)
                else:
                    nb_enemy.setdefault(move, set()).add(si)
                    
    return NbInfo(nb_libs, nb_friend, nb_enemy)

def correct_moves_mask(board, player, string_lib, nb_info):
    """
    # compute move distribution
    d = get_move_distribution(board,'B')
    # filter out incorrect intersections (set to 0)
    d = d * correct_moves_mask(board, 'b')
    """
    mask = np.zeros((board.side, board.side))
    
    for row in xrange(board.side):
        for col in xrange(board.side):
            move = (row, col)
            if board.get(row, col):
                assert not move in nb_info.liberties
                assert not move in nb_info.friends
                assert not move in nb_info.enemies
                #logging.debug("%s invalid: stone there"%(gomill.common.format_vertex((row, col))))
                continue
            # has liberties in nbhood
            if move in nb_info.liberties:
                mask[row][col] = 1
                #logging.debug("%s valid: liberty"%(gomill.common.format_vertex((row, col))))
                continue
            # has friendly string in nbhood
            if move in nb_info.friends:
                # who has different liberty than (row,col)
                if any( len(string_lib.liberties[si]) > 1 for si in nb_info.friends[move] ):
                    mask[row][col] = 1
                    #logging.debug("%s valid: alive friend"%(gomill.common.format_vertex((row, col))))
                    continue
            # enemy string in the nbhood, which we capture
            if move in nb_info.enemies:
                if any( len(string_lib.liberties[si]) == 1 for si in nb_info.enemies[move] ):
                    mask[row][col] = 1
                    #logging.debug("%s valid: killing enemy"%(gomill.common.format_vertex((row, col))))
                    continue
    return mask

def board2color_mask(board):
    black = np.zeros((board.side, board.side))
    white = np.zeros((board.side, board.side))
    empty = np.zeros((board.side, board.side))
    
    for row in xrange(board.side):
        for col in xrange(board.side):
            color = board.get(row, col)
            if color == 'b':
                black[row][col] = 1
            elif color == 'w':
                white[row][col] = 1
            else:
                empty[row][col] = 1
    return black, white, empty

def board2correct_move_mask(board, player):
    string_lib = board2string_lib(board)
    nb_info = analyze_nbhood(board, player, string_lib)
    return correct_moves_mask(board, player, string_lib, nb_info)

if __name__ == "__main__":
    def test_strings():
        with open("test_sgf/test1.sgf", 'r') as fin:
            game = gomill.sgf.Sgf_game.from_string(fin.read())
        
        board, movepairs = gomill.sgf_moves.get_setup_and_moves(game)
        for color, move in movepairs:
            if move:
                row, col = move
                board.play(row, col, color)
        
        colors, strings, liberties = board2strings(board)
       
        # XXX
        
        for row in xrange(board.side):
            print "%2d  " % row, 
            for col in xrange(board.side):
                pt = board.get(row, col)
                if pt == None:
                    print ".", 
                else:
                    print "%s" % '#' if pt == 'b' else 'o', 
            print
                
        print '    ', 
        for col in xrange(board.side):
            print "%s" % string.lowercase[col if col < 8 else col +1], 
        print
        
            
        for row in xrange(board.side):
            print "%2d  " % row, 
            for col in xrange(board.side):
                pt = board.get(row, col)
                if pt == None:
                    print " .", 
                else:
                    print "%2d" % strings[row, col], 
            print
                
        print '    ', 
        for col in xrange(board.side):
            print " %s" % string.lowercase[col if col < 8 else col +1], 
        print
        
        for i in xrange(max(strings.itervalues())):
            print i, len(liberties[i])

    def time_correctness():
        logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                            level=logging.DEBUG)
        import time
        with open("test_sgf/test2.sgf", 'r') as fin:
            game = gomill.sgf.Sgf_game.from_string(fin.read())
        
        tt0 = 0
        tt2 = 0
        tt3 = 0
        it = 0
        
        for i in xrange(10):
            board, movepairs = gomill.sgf_moves.get_setup_and_moves(game)
            for color, move in movepairs:
                if move:
                    row, col = move
                    board.play(row, col, color)
                    
                    s = time.clock()
                    sl = board2string_lib(board)
                    tt0 += time.clock() - s
                    
                    s = time.clock()
                    an = analyze_nbhood(board, 'w', sl)
                    tt3 += time.clock() - s
                    
                    s = time.clock()
                    m2 = correct_moves_mask(board, 'w', sl, an)
                    tt2 += time.clock() - s
                    
                    it += 1
        logging.debug("board2string   = %.3f, %.5f per one "%(tt0, tt0/it))
        logging.debug("analyze_nbhood = %.3f, %.5f per one "%(tt3, tt3/it))
        logging.debug("correct_moves  = %.3f, %.5f per one "%(tt2, tt2/it))
        logging.debug("it = %d"%(it))
        
    def test_correctness():
        logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                            level=logging.DEBUG)
        import time
        with open("test_sgf/correctness.sgf", 'r') as fin:
            game = gomill.sgf.Sgf_game.from_string(fin.read())
        
        board, movepairs = gomill.sgf_moves.get_setup_and_moves(game)
        for color, move in movepairs:
            if move:
                row, col = move
                board.play(row, col, color)
                
        sl = board2string_lib(board)
        an = analyze_nbhood(board, 'w', sl)
        mask = correct_moves_mask(board, 'w', sl, an)
        
        # see the correcness sgf
        enemystones = 'A3 B3 A18 G19 M11 D8 C8'
        ourstones = 'D14 B1 D1 A2 B2 T2 T19 M8 K6'
        suicides = 'A19 C19 T18 G18 D13 N16'
        captures = 'E8 T4 Q12'
        connects = 'K1 C1 A1 K7 T1'
        alone    = 'Q10 N5 N1 G1 G5 T14 T15'
        
        sets = [(enemystones, 0),
                (ourstones, 0),
                (suicides, 0),
                (captures, 1), 
                (connects, 1),
                (alone, 1)]
        
        for s, res in sets:
            moves = map(lambda vertex : gomill.common.move_from_vertex(vertex,
                                                                       board.side),
                        s.split())
            assert all (mask[row][col] == res for row, col in moves)
            
    time_correctness()


