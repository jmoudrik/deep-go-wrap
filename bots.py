import random
import numpy as np
import logging

import gomill
from gomill import common, boards, sgf, sgf_moves


class Bot(object):
    def genmove(self, state, color):
        """
        :return: a tuple (row, col), or None for a pass move
        """
        raise NotImplementedError


class DistributionBot(Bot):
    def gen_probdist(self, state, color):
        """
        Generates a probability distribution for the next move.
        
        :return: a numpy array of floats of shape (board.side, board.side), or None for pass
                 the array should be normalized to 1
        """
        raise NotImplementedError


class MaxDistributionBot(DistributionBot):
    """
    A simple bot which chooses next move to be the one with the biggest (therefore the name)
    probability. The probabilities are computed by gen_probdist().
    
    Never passes.
    """
    def gen_probdist(self, state, color):
        raise NotImplementedError
    def genmove(self, state, color):
        dist = self.gen_probdist(state, color)
        return np.unravel_index(np.argmax(dist), dist.shape)

    
class SamplingDistributionBot(DistributionBot):
    """
    A simple bot which randomly samples next move based on the moves' probability
    distribution, computed by gen_probdist().
    
    Never passes.
    """
    def gen_probdist(self, state, color):
        raise NotImplementedError
    def genmove(self, state, color):
        dist = self.gen_probdist(state, color)
        
        # choose an intersection with probability given by the dist
        coord = np.random.choice((state.board.side ** 2), 1, p=dist.ravel())[0]
        return (coord / state.board.side,  coord % state.board.side)
    
        
class RandomBot(Bot):
    def genmove(self, state, color):
        if last_move == 'pass':
            return None
        else:
            for i in xrange(10):
                row, col = np.random.choice(state.board.side, 2)
                ## TODO this might be incorrect move
                # but nobody will use the RandomBot anyway
                if not board.get(row, col):
                    return row, col
            return None
            
if __name__ == "__main__":
    def _genprobdist(state, color):
        a = np.random.random((state.board.side, state.board.side))
        x, y = np.random.choice(state.board.side, 2)
        # put the max on the x-y location
        # s.t. we get it often (66%)
        a[x][y] = state.board.side ** 2
        print("max: %d,%d"%(x, y))
        ret = a / a.sum()
        return ret
        
    def test_bot():
        #np.random.seed(10)
        bot = SamplingDistributionBot()
        #bot = MaxDistributionBot()
        bot.gen_probdist = _genprobdist
        
        from state import State
        
        b = gomill.boards.Board(19)
        print ("bot: %d,%d"%bot.genmove(State(b), 'w'))
        
    test_bot()
    