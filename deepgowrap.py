#!/usr/bin/env python
from __future__ import print_function

import logging
import re
import sys
import os

import gomill
from gomill import gtp_engine, gtp_states

from deepgo.players import *


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

    from deepgo import bot_deepcl

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

    player.name = "DeepCL CNN Bot, v0.1"

    # 4) make the GTP engine
    engine = make_engine(player)
    gomill.gtp_engine.run_interactive_gtp_session(engine)

def main_detlef():
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                        level=logging.DEBUG)

    from deepgo import bot_caffe
    # 0) you need to have caffe installed of course :-)
    import caffe

    # 1) set up caffe_net
    # you got to download the CNN from (as of January 2016)
    # http://physik.de/CNNlast.tar.gz
    # and change the path of files below
    caffe_net = caffe.Net('golast19.prototxt', 'golast.trained', 0)

    # 2) set up detlef distribution bot
    detlef_bot = bot_caffe.DetlefDistBot(caffe_net)

    # 3) make a player which plays the move with max probability
    #    and wrap it by GnuGo to pass correctly
    player =  WrappingGnuGoPlayer(DistWrappingMaxPlayer(detlef_bot))

    # change this to this if you do not have GnuGo installed (but the
    # bot will not pass nor resign...
    #player =  DistWrappingMaxPlayer(detlef_bot)

    player.name = "Detlef's 54% CNN Bot"

    # 4) make the GTP engine
    engine = make_engine(player)
    gomill.gtp_engine.run_interactive_gtp_session(engine)

if __name__ == "__main__":
    #main_random()
    #main_deepcl()
    main_detlef()

