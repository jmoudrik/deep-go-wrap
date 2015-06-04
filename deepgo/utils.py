import numpy as np
import logging
import os
import subprocess

import gomill
from gomill import common, boards

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
