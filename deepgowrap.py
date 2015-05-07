#!/usr/bin/env python
from __future__ import print_function

import logging
import re
import sys
from itertools import izip, ifilter, imap

import gomill
from gomill import gtp_engine, gtp_states

import players

from state import State

def make_engine(player):
    """Return a Gtp_engine_protocol which runs the specified player."""
    gtp_state = gtp_states.Gtp_state(move_generator=player.genmove)
    
    engine = gtp_engine.Gtp_engine_protocol()
    engine.add_protocol_commands()
    engine.add_commands(gtp_state.get_handlers())
    engine.add_commands(player.get_handlers())
    return engine

def main():
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                        level=logging.DEBUG)
    player = players.WrappingPassPlayer(players.RandomPlayer())
    
    engine = make_engine(player)
    gomill.gtp_engine.run_interactive_gtp_session(engine)

if __name__ == "__main__":
    main()

