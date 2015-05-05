import logging
import tempfile
import os
import subprocess

import gomill
from gomill import common, boards, sgf, sgf_moves

from bots import Bot

DEBUG = False

class WrappingPassBot(Bot):
    def __init__(self, bot):
        self.bot = bot
        
    def genmove(self, state, color):
        logging.debug("WrappingPassBot: enter")
        # pass if GnuGo tells us to do so
        if self.gnu_go_pass_check(state, color):
            logging.debug("WrappingPassBot: pass")
            return None
        else:
            logging.debug("WrappingPassBot: no pass, descend")
            return self.bot.genmove(state, color)
        
    def gnu_go_pass_check(self, state, color):
        assert isinstance(state.board, gomill.boards.Board) # for wingide code completion
        
        game = gomill.sgf.Sgf_game(size=state.board.side)
        gomill.sgf_moves.set_initial_position(game, state.board)
        node = game.get_root()
        node.set('KM', state.komi)
        node.set('PL', color)
        
        with tempfile.NamedTemporaryFile(delete=not DEBUG) as sgf_file:
            sgf_file.write(game.serialise())
            sgf_file.flush()
            gg_move = self.get_gnu_go_move(sgf_file.name, color)
        
        return gg_move.lower() == 'pass'
        
    def get_gnu_go_move(self, sgf_filename, color):
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
    