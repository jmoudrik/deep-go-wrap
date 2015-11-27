import numpy as np
import logging
import os
import subprocess

import gomill
from gomill import common, boards

def border_mark(boardsize=19):
    a = np.zeros((boardsize, boardsize), dtype='uint8')
    a[0,:]=1
    a[-1,:]=1
    a[:,0]=1
    a[:,-1]=1

    return a

def l1_distance(pt1, pt2):
    (x1, y1), (x2, y2) = pt1, pt2
    dx, dy = x1 - x2, y1 - y2
    return abs(dx) + abs(dy)

def sq_distance(pt1, pt2):
    (x1, y1), (x2, y2) = pt1, pt2
    dx, dy = x1 - x2, y1 - y2
    return dx**2 + dy**2

def l2_distance(pt1, pt2):
    return np.sqrt(sq_distance(pt1, pt2))

def gridcular_distance(pt1, pt2):
    (x1, y1), (x2, y2) = pt1, pt2
    dx, dy = abs(x1 - x2), abs(y1 - y2)
    return dx + dy + max(dx, dy)

def distances_from_pt(dist, point, boardsize=19):
    a = np.zeros((boardsize, boardsize), dtype='float32')

    for x in xrange(boardsize):
        for y in xrange(boardsize):
            a[x][y] = dist(point, (x,y))

    return a

def dist_stats(dist, top=3):
    def ind2pt(ind):
        return ind / side, ind % side

    def format_move_prob((row, col), prob):
        return "%s  %.6f"%( gomill.common.format_vertex((row, col)), prob)

    assert dist.shape[1] == dist.shape[0]
    side = dist.shape[0]

    sort_ind = np.argsort(dist.ravel())
    ret = []

    # some statistics
    row, col = ind2pt(sort_ind[0])
    ret.append("mean: \t%.6f"%(np.mean(dist)))
    ret.append("stddev: %.6f"%(np.std(dist)))
    #ret.append("min: \t" + format_move_prob((row, col), dist[row][col]))

    s = 0.0
    # list of # top moves
    for num, i in enumerate(reversed(sort_ind[-top:])):
        row, col = ind2pt(i)
        prob = dist[row][col]
        s += prob
        ret.append("%d: \t"%(num+1) + format_move_prob((row, col), prob))

    ret.append("top %d moves cover: %.2f %%" % (top, 100* s))


    return "\n".join(ret)

def raw_history(board, history):
    """
    History counter, returns array of time since the last stone
    at location was played.
    Watch out, this should be masked by valid moves, since stones that
    are taken out are still present!

    In case of multiple stones being played at one place (ko, playing under stones, ...)
    the last one is remembered

    last move = 1
    first one = #of moves
    empty points = #of moves +1

    """
    a = np.zeros((board.side, board.side))
    time = 0
    for color, move in history:
        time += 1
        # first move should have 1
        a[move] = time

    # The first one now will later be last!
    return time + 1 - a

def get_gnu_go_response(sgf_filename, color):
    """
    returns None if we could not get gnugo move.
    otw, returns raw gnugo response "PASS","D9",...
    """
    logging.debug("get_gnu_go_move(sgf_filename='%s', color='%s')"%(sgf_filename,
                                                                    color))
    GTP_CMD = "loadsgf %s\ngenmove %s\n"% (sgf_filename, color)
    p = subprocess.Popen(['gnugo', '--level',  '1', '--mode', 'gtp'],
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)

    stdout, stderr = p.communicate(GTP_CMD)


    if stderr:
        logging.warn("GnuGo stderr: '%s'"%stderr)

    # split & filter out empty strings
    responses = [ tok.strip() for tok in stdout.split('\n') if tok ]

    failed = False
    for resp in responses:
        if resp[0] == '?':
            logging.warn("GnuGo error: '%s'"%resp)
            failed = True

    sign, move = responses[-1].split()
    # success
    if sign == '=':
        logging.debug("GnuGo would play %s"%move)
        return move

    if failed:
        logging.warn("Could not get GnuGo move.")
        return None

if __name__ ==  "__main__":
    def test_stat():
        a = np.random.random((3, 3))
        a[1][1] = 0
        a = a / a.sum()

        print dist_stats(a)
