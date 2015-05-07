from __future__ import print_function

import logging
import tempfile
import os
import subprocess
import numpy as np

import gomill
from gomill import common, boards, sgf, sgf_moves, gtp_states

"""
Basic Player / Bot objects;

Player should be gomill compatible envelope which actually generates
    moves, resigns, ..
    
Bot should be the object that actually does the core work, e.g. computing
Move probability, ..
"""

class Player(object):
    def __init__(self):
        self.handlers = { 'name' : self.handle_name,
                          'quit' : self.handle_quit }
    def genmove(self, state, player):
        """
        :returns: gomill.Move_generator_result
        """
        raise NotImplementedError
    def handle_name(self, args):
        return self.__class__
    def handle_quit(self, args):
        pass
    def get_handlers(self):
        return self.handlers

class DistWrappingMaxPlayer(Player):
    """
    A simple wrapping bot which chooses next move to be the one with the biggest (therefore the name)
    probability. The probabilities are computed by the wrapped bot's gen_probdist().
    
    Never passes.
    """
    def __init__(self, bot):
        super(DistWrappingMaxPlayer,  self).__init__()
        self.bot = bot
    def genmove(self, state, player):
        dist = self.bot.gen_probdist(state, player)
        move = np.unravel_index(np.argmax(dist), dist.shape)
        
        result = gtp_states.Move_generator_result()
        result.move = move
        return result
    def handle_quit(self, args):
        self.bot.close()

    
class DistWrappingSamplingPlayer(Player):
    """
    A simple wrapping bot which randomly samples next move based on the moves' probability
    distribution, computed by the wrapped bot's gen_probdist().
    
    Never passes.
    """
    def __init__(self, bot):
        super(DistWrappingSamplingPlayer,  self).__init__()
        self.bot = bot
    def genmove(self, state, player):
        dist = self.bot.gen_probdist(state, player)
        # choose an intersection with probability given by the dist
        coord = np.random.choice((state.board.side ** 2), 1, p=dist.ravel())[0]
        move = (coord / state.board.side,  coord % state.board.side)
    
        result = gtp_states.Move_generator_result()
        result.move = move
        return result
    def handle_quit(self, args):
        self.bot.close()
    
        
class RandomPlayer(Player):
    def genmove(self, state, player):
        result = gtp_states.Move_generator_result()
        # pass
        if state.move_history and not state.move_history[-1].move:
            result.pass_move = True
            return result
        else:
            for i in xrange(10):
                row, col = np.random.choice(state.board.side, 2)
                ## TODO this might be incorrect move
                # but nobody will use the RandomPlayer anyway
                if not state.board.get(row, col):
                    result.move =  (row, col)
                    return result
            result.resign = True
            return result
    

def get_gnu_go_response(sgf_filename, color):
    """
    returns None if we could not get gnugo move.
    otw, returns raw gnugo response "PASS","D9",...
    """
    logging.debug("get_gnu_go_move(sgf_filename='%s', color='%s')"%(sgf_filename,
                                                                    color))
    GTP_CMD = "loadsgf %s\ngenmove %s\n"% (sgf_filename, color)
    p = subprocess.Popen(['gnugo', '--mode', 'gtp'],
                         stdin=subprocess.PIPE, 
                         stdout=subprocess.PIPE)
    stdout = p.communicate(GTP_CMD)[0]
        
    
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

class WrappingPassPlayer(Player):
    def __init__(self, player):
        super(WrappingPassPlayer,  self).__init__()
        self.player = player
        
    def genmove(self, state, color):
        result = gtp_states.Move_generator_result()
    
        logging.debug("WrappingPassBot: enter")
        # pass if GnuGo tells us to do so
        if self.gnu_go_pass_check(state, color):
            logging.debug("WrappingPassBot: pass")
            result.pass_move = True
            return result
        else:
            logging.debug("WrappingPassBot: no pass, descend")
            return self.player.genmove(state, color)
        
    def handle_quit(self, args):
        self.player.handle_quit(args)
        
    def gnu_go_pass_check(self, state, color):
        assert isinstance(state.board, gomill.boards.Board) # for wingide code completion
        
        game = gomill.sgf.Sgf_game(size=state.board.side)
        gomill.sgf_moves.set_initial_position(game, state.board)
        node = game.get_root()
        node.set('KM', state.komi)
        node.set('PL', color)
        
        with tempfile.NamedTemporaryFile() as sgf_file:
            sgf_file.write(game.serialise())
            sgf_file.flush()
            gg_move = get_gnu_go_response(sgf_file.name, color)
        
        return gg_move.lower() == 'pass'
    
    
class DistributionBot(object):
    def gen_probdist(self, state, player):
        """
        Generates a probability distribution for the next move.
        
        :return: a numpy array of floats of shape (board.side, board.side), or None for pass
                 the array should be normalized to 1
        """
        raise NotImplementedError
    def close(self):
        """Called upon exit, to allow for resource freeup."""
        pass

        
class RandomDistBot(DistributionBot):
    def gen_probdist(self, state, player):
        a = np.random.random((state.board.side, state.board.side))
        x, y = np.random.choice(state.board.side, 2)
        # put the max on the x-y location
        # s.t. we get it often (66%)
        a[x][y] = state.board.side ** 2
        ret = a / a.sum()
        
        logging.debug("max at: %d,%d"%(x, y))
        return ret
            
if __name__ == "__main__":
        
    def test_bot():
        logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                        level=logging.DEBUG)
        player = DistWrappingSamplingPlayer(RandomDistBot())
        
        class State:
            pass
        s = State()
        
        b = gomill.boards.Board(19)
        s.board = b
        logging.debug("bot: %s"% repr(player.genmove(s, 'w').move))
        
    test_bot()
    