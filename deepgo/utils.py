import numpy as np
import logging

import gomill
from gomill import common, boards

import go_strings

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
    colors, strings, liberties = go_strings.board2strings(board)
    
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
            for nb in go_strings.iter_nbhs(board, (row, col)):
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


if __name__ ==  "__main__":
    def test_stat():
        a = np.random.random((3, 3))
        a[1][1] = 0
        a = a / a.sum()

        print dist_stats(a)
        
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


