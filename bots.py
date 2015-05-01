import random
import numpy as np

class Bot(object):
    def genmove(self, board, color, last_move, ko_forbidden_move, komi):
        """
        :return: a tuple (row, col), or None for a pass move
        """
        raise NotImplementedError
    
class DistributionBot(Bot):
    def gen_protodist(self, board, color, last_move, ko_forbidden_move, komi):
        """
        Generates a "proto distribution" for the next move. The difference between
        this and the previous method is that this proto distribution need not
        be normalized, so this can be e.g. number of MCTS playouts for each move.
        
        :return: a numpy array of floats of shape (board.side, board.side), or None for pass
        """
        raise NotImplementedError
    def gen_probdist(self, board, color, last_move, ko_forbidden_move, komi):
        """
        Generates a probability distribution for the next move.
        
        :return: a numpy array of floats of shape (board.side, board.side), or None for pass
                 the array should be normalized to 1
        """
        raise NotImplementedError
    
class SimpleDistributionBot(DistributionBot):
    def gen_protodist(self, board, color, last_move, ko_forbidden_move, komi):
        raise NotImplementedError
    def gen_probdist(self, *args):
        proto_dist = self.gen_protodist(*args)
        return proto_dist / np.sum(proto_dist)            
    def genmove(self, board, color, last_move, ko_forbidden_move, komi):
        if last_move == 'pass':
            return None
        dist = self.gen_probdist(board, color, last_move, ko_forbidden_move, komi)
        return np.unravel_index(np.argmax(dist), a.shape)
    
class RandomBot(Bot):
    def genmove(self, board, color, last_move, ko_forbidden_move, komi):
        ## TODO smarter pass
        if last_move == 'pass':
            return None
        ## TODO smarter play
        else:
            for i in xrange(10):
                row, col = random.randint(0, board.side-1), random.randint(0, board.side-1)
                if not board.get(row, col):
                    return row, col
            return None
