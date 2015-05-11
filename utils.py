import numpy as np

import gomill
from gomill import common, boards


def empty_board_mask(board):
    a = np.ones((board.side, board.side))
    for row in xrange(board.side):
        for col in xrange(board.side):
            if board.get(row, col):
                a[row][col] = 0
    return a

def dist_stats(dist, top=3):
    assert dist.shape[1] == dist.shape[0]
    side = dist.shape[0]
    
    best_top_ind = np.argsort(dist.ravel())[-top:]
    ret = []
    sum_prob = 0
    for num, i in enumerate(reversed(best_top_ind)):
        row, col = i / side, i % side
        prob = 100 * dist[row][col]
        sum_prob += prob
        ret.append("%d: %s  %.1f%%"%(num + 1,
                                      gomill.common.format_vertex((row, col)),
                                      prob))
                   
    #ret.append("Top %d sum: %.1f%%"%(top, sum_prob))
                   
    return "\n".join(ret)


if __name__ ==  "__main__":
    def test():
        a = np.random.random((3, 3))
        a[1][1] = 0
        a = a / a.sum()
        
        print dist_stats(a)
        
        
    
    test()
        
    