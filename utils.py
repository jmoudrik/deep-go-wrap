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
    ret.append("min: \t" + format_move_prob((row, col), dist[row][col]))

    # list of # top moves
    for num, i in enumerate(reversed(sort_ind[-top:])):
        row, col = ind2pt(i)
        prob = dist[row][col]
        ret.append("%d: \t"%(num+1) + format_move_prob((row, col), prob))


    return "\n".join(ret)


if __name__ ==  "__main__":
    def test():
        a = np.random.random((3, 3))
        a[1][1] = 0
        a = a / a.sum()

        print dist_stats(a)



    test()


