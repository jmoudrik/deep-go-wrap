from itertools import product
import string

import gomill
from gomill import boards, sgf, sgf_moves

#NBCOORD_DIAG = tuple(product((1, -1), (1, -1)))

NBCOORD = ((-1,0), (1,0), (0,1), (0,-1))

def board2strings(board):
    """
    Divides board into strings and computes sets of their liberties.
    Takes O(N) time and space, where N is the number of occupied points.
    Algorithm is a simple dfs.
    """
    
    def coord_onboard(row, col):
        return row >= 0 and col >= 0 and row < board.side and col < board.side
    def iter_nbhs((row, col)):
        for dx, dy in NBCOORD:
            nbx, nby = row + dx, col + dy
            if coord_onboard(nbx, nby):
                yield nbx, nby
    
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
            for nb in iter_nbhs(n):
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

if __name__ == "__main__":
    with open("test_sgf/test.sgf", 'r') as fin:
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



