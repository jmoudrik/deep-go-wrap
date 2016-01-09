import tempfile
import subprocess
import os
import logging
import array
import numpy as np
import time

import caffe

from players import DistributionBot, DistWrappingMaxPlayer
import cubes
from state import gomill_gamestate2state, State

class DetlefDistBot(DistributionBot):
    """
    CNN as kindly provided by Detlef Schmicker. See
    http://computer-go.org/pipermail/computer-go/2015-December/008324.html


    The net should (as of January 2016) be available here:
    http://physik.de/CNNlast.tar.gz

    """
    def __init__(self, caffe_net):
        super(DetlefDistBot,  self).__init__()
        self.caffe_net = caffe_net

    def gen_probdist_raw(self, game_state, player):
        cube = cubes.get_cube_detlef(gomill_gamestate2state(game_state), player)
        cube = cube.reshape( (1,) + cube.shape)

        logging.debug("%s sending data of shape=%s"%(self, cube.shape))

        resp = self.caffe_net.forward_all(**{'data':cube})['ip']
        logging.debug("%s read response of shape=%s"%(self, resp.shape))

        # FIXME update, 128 output channels is detlef's mistake :-)
        resp = resp.reshape((128, game_state.board.side, game_state.board.side))
        tot = resp.sum()
        ret = resp[0]
        logging.debug("%s trimming channelstook off %.3f %%"%(self,
                                                                100 * (tot - ret.sum())/tot))

        return ret / ret.sum()

if __name__ == "__main__":
    def test_bot():
        import gomill
        import rank

        logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                            level=logging.DEBUG)

        # substitute with your own
        caffe_net = caffe.Net('golast19.prototxt', 'golast.trained', 0)

        player = DistWrappingMaxPlayer(DetlefDistBot(caffe_net))

        class GameState:
            pass
        s = GameState()
        s.board = gomill.boards.Board(19)
        s.ko_point = None
        s.move_history = []

        print player.genmove(s, 'b').move

    test_bot()
