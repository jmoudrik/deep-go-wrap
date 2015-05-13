#!/usr/bin/env python
from __future__ import print_function

import logging
import re
import sys

import gomill
from gomill import gtp_engine, gtp_states

from players import * 

from state import State
import bot_deepcl

def make_engine(player):
    """Return a Gtp_engine_protocol which runs the specified player."""
    gtp_state = gtp_states.Gtp_state(move_generator=player.genmove)
    
    engine = gtp_engine.Gtp_engine_protocol()
    engine.add_protocol_commands()
    engine.add_commands(gtp_state.get_handlers())
    engine.add_commands(player.get_handlers())
    return engine

def main_random():
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                        level=logging.DEBUG)
    # player def
    # player = players.WrappingPassPlayer(players.RandomPlayer())
    player = DistWrappingMaxPlayer(RandomDistBot())
    
    # player => engine => RUN
    engine = make_engine(player)
    gomill.gtp_engine.run_interactive_gtp_session(engine)

def main_deepcl():
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                        level=logging.DEBUG)
    
    DCL_PATH = '/home/jm/prj/DeepCL/'
    deepcl_io = bot_deepcl.DeepCL_IO(os.path.join(DCL_PATH, 'build/deepclexec'), options={
        #'dataset':'kgsgo',
        'weightsfile': os.path.join(DCL_PATH, "build/weights.dat"), 
        'datadir': os.path.join(DCL_PATH, 'data/kgsgo'),
        # needed to establish normalization parameters
        'trainfile': 'kgsgo-train10k-v2.dat',})
    deepcl_bot = bot_deepcl.DeepCLDistBot(deepcl_io)

    player = DistWrappingMaxPlayer(deepcl_bot)
    engine = make_engine(player)
    gomill.gtp_engine.run_interactive_gtp_session(engine)

if __name__ == "__main__":
    #main_random()
    main_deepcl()

