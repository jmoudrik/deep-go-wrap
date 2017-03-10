from itertools import product
import string
import logging
import numpy as np
from collections import namedtuple

import gomill
from gomill import boards, sgf, sgf_moves, ascii_boards

import utils

NBCOORD_DIAG = tuple(product((1, -1), (1, -1)))
NBCOORD = ((-1,0), (1,0), (0,1), (0,-1))

# StringLib.string = {}
# coord => string number
# StringLib.liberties = {}
# string number => set of liberties coords
# StringLib.liberties_nb_count = {}
# liberty coor => number of neighboring stones
StringLib = namedtuple('StringLib', 'string liberties liberties_nb_count')

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

def coord_count_edges(board, (row, col)):
    return sum((row == 0, col == 0, row == board.side - 1, col == board.side -1))

def iter_nbhs(board, (row, col)):
    for dx, dy in NBCOORD:
        nbx, nby = row + dx, col + dy
        if 0 <= nbx < board.side and 0 <= nby < board.side:
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
    # liberty => set of nbs
    lib_nb_cnt = {}

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
                    lib_nb_cnt[nb] = lib_nb_cnt.get(nb, 0) + 1
                    liberties.setdefault(i, set()).add(nb)
                    continue
                if nb in visited:
                    # better be safe
                    assert ((colors[nb] != colors[n])
                            or visited[nb] == visited[n])
                    continue
                if colors[nb] == colors[n]:
                    fringe.append(nb)

    return StringLib(visited, liberties, lib_nb_cnt)

def board2dist_from_stones(board, player, maxdepth=4):
    """
    For each point, compute distance to the closest B or W stone.
    """
    # max L1 distance on goban is between (0,0) and (18,18) which is 36
    # so 2*19 is good inf
    inf = board.side * 2

    def bfs(a, fringe, depth=0):
        if depth > maxdepth:
            return a

        f = set()
        for pt in fringe:
            a[pt] = depth
            for nb in iter_nbhs(board, pt):
                if a[nb] == inf and nb not in fringe:
                    f.add(nb)
        if f:
            bfs(a, f, depth+1)
        return a

    # fringes
    us, them = set(), set()

    for color, pt in board.list_occupied_points():
        if color == player:
            us.add(pt)
        else:
            them.add(pt)

    def gd(fringe):
        d = np.full((board.side, board.side), inf, dtype='uint8')
        return bfs(d, fringe)

    return gd(us), gd(them)

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
    mask = np.zeros((board.side, board.side), dtype='uint8')

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

def board2color_mask(board, player):
    empty = np.zeros((board.side, board.side), dtype='uint8')
    friend = np.zeros((board.side, board.side), dtype='uint8')
    enemy = np.zeros((board.side, board.side), dtype='uint8')

    for row in xrange(board.side):
        for col in xrange(board.side):
            color = board.get(row, col)
            if color is None:
                empty[row][col] = 1
            elif color == player:
                friend[row][col] = 1
            else:
                enemy[row][col] = 1
    return empty, friend, enemy

def liberties_count(board, string_lib):
    liberties = np.zeros((board.side, board.side))
    for (row, col), si in string_lib.string.iteritems():
        liberties[row][col] = len(string_lib.liberties[si])

    return liberties

def board2correct_move_mask(board, player):
    string_lib = board2string_lib(board)
    nb_info = analyze_nbhood(board, player, string_lib)
    return correct_moves_mask(board, player, string_lib, nb_info)

def construct_closeset(boardsize, depth):
    """
    Ugly & slow, only run once
    """
    ret = []
    for row in xrange(boardsize):
        ret.append([])
        for col in xrange(boardsize):
            # l1 distance
            cand=[]
            for x, y in product(xrange(row-depth, row+depth +1),
                                xrange(col-depth, col+depth +1)):
                if 0 <= x < boardsize and 0 <= y < boardsize:
                    l1 = utils.l1_distance((row, col), (x,y))
                    if 0 <= l1 <= depth:
                        cand.append((l1,x,y))
            # s.t. neigbors are ordered by the distance
            cand.sort()
            ret[-1].append((np.array([c[0] for c in cand]),
                            np.array([c[1] for c in cand]),
                            np.array([c[2] for c in cand])))

    return ret

def npclose(a, empty, closeset, verbose=False):
    inf = a.shape[0] * 2
    ret = np.full(a.shape, inf)
    it = np.nditer(empty, flags=['multi_index'])
    while not it.finished:
        x,y = it.multi_index

        dist,xs,ys = closeset[x][y]
        arg = a[xs,ys].argmax()
        if a[xs[arg],ys[arg]]:
            ret[x][y] = dist[arg]

        it.iternext()

    return ret

def lib_nbs_to_lib_count(board, liberties_nb_count):
    """
    returns array of number of empty intersection around empty intersection.
    invalid for coords where a stone is.

    essentialy counts liberties of liberties
    """
    lib_counts = np.zeros((board.side, board.side), dtype='uint8')
    for row in xrange(board.side):
        for col in xrange(board.side):
            lib = row, col
            lib_counts[row][col] = 5 - liberties_nb_count.get(lib, 0) - coord_count_edges(board, lib)

    return lib_counts


if __name__ == "__main__":
    def print_board(board):
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


    def test_strings():
        with open("test_sgf/test1.sgf", 'r') as fin:
            game = gomill.sgf.Sgf_game.from_string(fin.read())

        board, movepairs = gomill.sgf_moves.get_setup_and_moves(game)
        for color, move in movepairs:
            if move:
                row, col = move
                board.play(row, col, color)

        sl = board2string_lib(board)
        libcnt = lib_nbs_to_lib_count(board, sl.liberties_nb_count)

        # XXX
        print_board(board)

        for row in xrange(board.side):
            print "%2d  " % row,
            for col in xrange(board.side):
                pt = board.get(row, col)
                if pt == None or True:
                    print "%d"%(libcnt[row][col]),
                else:
                    print "%s" % '.' if pt == 'b' else '_',
            print

        print '    ',
        for col in xrange(board.side):
            print "%s" % string.lowercase[col if col < 8 else col +1],
        print

        #for i in xrange(max(strings.itervalues())):
        #    print i, len(liberties[i])

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

    def test_libdist():
        b = gomill.boards.Board(3)
        moves = [((0,1), 'b'),
                ((2,0), 'w'),
                ((1,2), 'b'),
                ((1,1), 'b')]

        b2 = gomill.boards.Board(3)
        for (x,y),col in moves:
            b.play(x,y, col)
            b2.play(2-x,y, col) # mirror s.t. we have the same coords
                                # with numpy arrays

        db, dw = board2dist_from_stones(b, 'w')

        #closest = db
        print gomill.ascii_boards.render_board(b2)
        print db < dw
        print dw < db

    def time_bfs():
        logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                            level=logging.DEBUG)
        import time
        with open("../test_sgf/test2.sgf", 'r') as fin:
            game = gomill.sgf.Sgf_game.from_string(fin.read())

        closeset = construct_closeset(19, 4)

        for i in xrange(1):
            board, movepairs = gomill.sgf_moves.get_setup_and_moves(game)

            num = 0
            for color, move in movepairs:
                if move:
                    num+=1
                    row, col = move
                    board.play(row, col, color)


                    empty, friend, enemy = board2color_mask(board, 'w')
                    cb = npclose(friend, empty, closeset)

                    break
                    cw = npclose(enemy, empty, closeset)

                    continue
                    db, dw = board2dist_from_stones(board, 'w', 4)

                    r1 = (db < dw) != (cb < cw)
                    r2 = (db > dw) != (cb > cw)

                    if r1.any() or r2.any():
                        print gomill.ascii_boards.render_board(board)
                        print "dist us"
                        print db
                        print "dist them"
                        print dw
                        print "nclose us"
                        print cb
                        print "nclose them"
                        print cw
                        print num
                        if r1.any():
                            i = np.argmax(r1)
                            i = np.unravel_index([i],db.shape)
                            i = i[0][0],i[1][0]
                            print "r1[%s] == True: %d < %d = %s, %d < %d = %s"%(i,
                                                    db[i], dw[i], (db < dw)[i],
                                                    cb[i], cw[i], (cb < cw)[i])
                        if r2.any():
                            i = np.argmax(r2)
                            i = np.unravel_index([i],db.shape)
                            i = i[0][0],i[1][0]
                            print "r2[%s] == True: %d > %d = %s, %d > %d = %s"%(i,
                                                    db[i], dw[i], (db > dw)[i],
                                                    cb[i], cw[i], (cb > cw)[i])
                        assert False

    def run():
        closeset = construct_closeset(19, 4)
        empty, friend, enemy = board2color_mask(gomill.boards.Board(side=19), 'w')
        for a in xrange(2000):
            cb = npclose(friend, empty, closeset)

    import cProfile
    #cProfile.run("run()")
#    cProfile.run("time_bfs()")

    #time_bfs()
    #test_libdist()
    test_strings()

