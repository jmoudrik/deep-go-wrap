from itertools import product
import string

import gomill
from gomill import boards, sgf, sgf_moves

#NBCOORD_DIAG = tuple(product((1, -1), (1, -1)))

NBCOORD = ((-1,0), (1,0), (0,1), (0,-1))

def coord_onboard(board, (row, col)):
    return row >= 0 and col >= 0 and row < board.side and col < board.side

def iter_nbhs(board, (row, col)):
    for dx, dy in NBCOORD:
        nbx, nby = row + dx, col + dy
        if coord_onboard(board, (nbx, nby)):
            yield nbx, nby

def board2strings(board):
    """
    Divides board into strings and computes sets of their liberties.
    Takes O(N) time and space, where N is the number of occupied points.
    Algorithm is a simple dfs.
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
                    
    return colors, visited, liberties

def empty_board_mask(board):
    mask = np.ones((board.side, board.side))
    for row in xrange(board.side):
        for col in xrange(board.side):
            if board.get(row, col):
                a[row][col] = 0
    return mask

def correct_moves_mask(board, player):
    """
    # compute move distribution
    d = get_move_distribution(board,'B')
    # filter out incorrect intersections (set to 0)
    d = d * correct_moves_mask(board, 'b')
    """
    colors, strings, liberties = board2strings(board)
    
    mask = np.zeros((board.side, board.side))
    # fill in correct moves
    for row in xrange(board.side):
        for col in xrange(board.side):
            # move is incorrect if it is nonempty
            if board.get(row, col):
                #logging.debug("%s invalid: stone there"%(gomill.common.format_vertex((row, col))))
                continue
            # correct if either we have a liberty in the neighborhood
            # or we dont AND (we connect with friends who do OR
            #                 we capture something)
            for nb in iter_nbhs(board, (row, col)):
                row_nb, col_nb = nb
                nb_color = board.get(row_nb, col_nb)
                # liberty
                if not nb_color:
                    mask[row][col] = 1
                    #logging.debug("%s valid: liberty"%(gomill.common.format_vertex((row, col))))
                    break
                # neighbor is not a liberty, so it is a stone, so
                # it belongs to a string and has some liberties
                libs = liberties[strings[nb]]
                if nb_color == player:
                    # connect to a friend
                    if len(libs - set([(row, col)])):
                        mask[row][col] = 1
                        #logging.debug("%s valid: alive friend"%(gomill.common.format_vertex((row, col))))
                        break
                else:
                    # capture some enemy stones
                    if len(libs) == 1:
                        # our move must be the last liberty
                        #logging.debug("%s valid: killing enemy"%(gomill.common.format_vertex((row, col))))
                        assert libs == set([(row, col)])
                        mask[row][col] = 1
                        break
                    
            if mask[row][col] == 0:
                pass
                #logging.debug("%s invalid: would be suicide"%(gomill.common.format_vertex((row, col))))
                
    return mask

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

    def test_correctness():
        logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                            level=logging.DEBUG)
        with open("test_sgf/correctness.sgf", 'r') as fin:
            game = gomill.sgf.Sgf_game.from_string(fin.read())
        
        board, movepairs = gomill.sgf_moves.get_setup_and_moves(game)
        for color, move in movepairs:
            if move:
                row, col = move
                board.play(row, col, color)
                
        correct_moves_mask(board, 'w')

    test_correctness()


