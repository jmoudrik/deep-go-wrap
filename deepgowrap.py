#!/usr/bin/env python
from __future__ import print_function

import logging
import re
import sys
import os

import gomill
from gomill import gtp_engine, gtp_states

from deepgo.players import *

from deepgo import bot_deepcl

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
    # player = players.WrappingGnuGoPlayer(players.RandomPlayer())
    player = DistWrappingMaxPlayer(RandomDistBot())

    # player => engine => RUN
    engine = make_engine(player)
    gomill.gtp_engine.run_interactive_gtp_session(engine)

def main_deepcl():
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                        level=logging.DEBUG)

    ## DCL_PATH = '/PATH/TO/YOUR/DeepCL'
    DCL_PATH = '/home/jm/prj/DeepCL/'
    # 1) set up io
    deepcl_io = bot_deepcl.DeepCL_IO(os.path.join(DCL_PATH, 'build/predict'), options={
        ## set up your weights file
        'weightsfile': os.path.join(DCL_PATH, "build/weights.dat"),
        'outputformat': 'binary',
            })
    # 2) set up deepcl distribution bot
    deepcl_bot = bot_deepcl.DeepCLDistBot(deepcl_io)

    # 3) make a player which plays the move with max probability
    #    and wrap it by GnuGo to pass correctly
    player =  WrappingGnuGoPlayer(DistWrappingMaxPlayer(deepcl_bot))
    
    # 4) make the GTP engine
    engine = make_engine(player)
    gomill.gtp_engine.run_interactive_gtp_session(engine)

if __name__ == "__main__":
    #main_random()
    main_deepcl()

