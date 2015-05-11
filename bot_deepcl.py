from __future__ import print_function

import tempfile
import subprocess
import os
import logging
import array
import numpy as np

from players import DistributionBot, DistWrappingMaxPlayer
import gomill

import go_strings

def get_plane_cube_v2(state, player):
    """Currently supports v2 version compatible planes
    https://github.com/hughperkins/kgsgo-dataset-preprocessor
    """
    colors, strings, liberties = go_strings.board2strings(state.board)

    def is_our_stone((row, col)):
        return state.board.get(row, col) == player
    def is_enemy_stone((row, col)):
        return state.board.get(row, col) == gomill.common.opponent_of(player)

    plane_functions = [ (lambda pt : (is_our_stone(pt) and len(liberties[strings[pt]]) == 1)),
                        (lambda pt : (is_our_stone(pt) and len(liberties[strings[pt]]) == 2)),
                        (lambda pt : (is_our_stone(pt) and len(liberties[strings[pt]]) >= 3)),
                        (lambda pt : (is_enemy_stone(pt) and len(liberties[strings[pt]]) == 1)),
                        (lambda pt : (is_enemy_stone(pt) and len(liberties[strings[pt]]) == 2)),
                        (lambda pt : (is_enemy_stone(pt) and len(liberties[strings[pt]]) >= 3)),
                        (lambda pt : (pt == state.ko_point)) ]

    def iterate_board(board):
        for row in xrange(board.side):
            for col in xrange(board.side):
                yield row, col

    # cube composed of 4-byte floats, with value of either 0, or 255
    cube_array = array.array('f')
    for planefc in plane_functions:
        for pt in iterate_board(state.board):
            cube_array.append(planefc(pt) * 255)
    return cube_array

class DeepCL_IO(object):
    def __init__(self,
                 deepclexec_path,
                 options={
                     "dataset" : "kgsgo", # needed for normalization
                     # "weightsFile": "weights.dat", # default value
                     # see 'deepclexec -h' for other options
                     # CAVEEAT: normalization has to be set up the same
                     # as when the CNN was trained
                 }):
        # DeepCL works with 4 byte floats, so we need to ensure we have
        # the same size, if this fails, we could probably reimplement it
        # using struct module
        self.itemsize = 4
        a = array.array('f')
        assert a.itemsize == self.itemsize
        
        self.deepclexec_path = deepclexec_path

        for res_opt in ['inputfile', 'outputfile']:
            if res_opt in options:
                logging.warn("DeepCL_IO: '%s' option is reserved, overriding."%res_opt)

        # first create the named pipes for IO
        self.tempdir = tempfile.mkdtemp()

        self.pipe_fn_to = os.path.join(self.tempdir, "PIPE_to")
        self.pipe_fn_from = os.path.join(self.tempdir, "PIPE_from")

        options['inputfile'] = self.pipe_fn_to
        options['outputfile'] = self.pipe_fn_from

        os.mkfifo(self.pipe_fn_to)
        os.mkfifo(self.pipe_fn_from)

        self.p = subprocess.Popen([deepclexec_path] + [ "%s=%s"%(k, v) for k, v in options.iteritems() ],
                                  stdin=None,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT)

        # this might seem like a too verbose piece of code, but
        # unfortunately, open() hangs if the other side is not opened,
        # so it is good to see this in log in the case...
        logging.debug("Setting up pipe: "+ self.pipe_fn_to)
        self.pipe_to = open(self.pipe_fn_to, 'wb')
        logging.debug("Setting up pipe: "+ self.pipe_fn_from)
        self.pipe_from = open(self.pipe_fn_from, 'rb')
        logging.debug("Pipes set up.")
        
    def gather_sub_logs(self):
        logging.debug("Waiting for subprocess to end...")
        stdout, stderr =  self.p.communicate()
        logging.debug("stdout:\n"+str(stdout) +"\n")
        logging.debug("stderr:\n"+str(stderr) +"\n")

    def close_pipes(self):
        self.pipe_to.close()
        self.pipe_from.close()
        
    def close(self):
        self.close_pipes()
        self.p.terminate()
        
        os.unlink(self.pipe_fn_to)
        os.unlink(self.pipe_fn_from)
        os.rmdir(self.tempdir)

    def write_cube(self, cube_array):
        cube_array.tofile(self.pipe_to)
        self.pipe_to.flush()

    def read_response(self, side):
        a = array.array('f')
        a.fromfile(self.pipe_from, side*side)
        return a

class DeepCLDistBot(DistributionBot):
    def __init__(self, deepcl_io):
        self.deepcl_io = deepcl_io

    def gen_probdist(self, state, player):
        cube = get_plane_cube_v2(state, player)
        
        try:
            logging.debug("Sending data cube of size %d B to CNN."%(self.deepcl_io.itemsize * len(cube)))
            self.deepcl_io.write_cube(cube)
            
            logging.debug("Reading response from CNN...")
            response = self.deepcl_io.read_response(state.board.side)
        except:
            self.deepcl_io.close_pipes()
            self.deepcl_io.gather_sub_logs()
            raise
        
        logging.debug("Got response of size %d B"%(self.deepcl_io.itemsize * len(response)))
        
        image = np.frombuffer(response, np.float32)
        return image.reshape((state.board.side, state.board.side))

    def close(self):
        self.deepcl_io.close()


if __name__ == "__main__":
    def test_bot():
        logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                            level=logging.DEBUG)

        DCL_PATH = '/home/jm/prj/DeepCL/build/'
        deepcl_io = DeepCL_IO(DCL_PATH + 'deepclexec', options={
            #'dataset':'kgsgo',
            'weightsfile': DCL_PATH + "weights.dat",
            'datadir': '/home/jm/prj/DeepCL/data/kgsgo',
            'trainfile': 'kgsgo-train10k-v2.dat',})

        player = DistWrappingMaxPlayer(DeepCLDistBot(deepcl_io))

        class State:
            pass
        s = State()

        b = gomill.boards.Board(19)
        s.board = b
        s.ko_point = None
        logging.debug("bot: %s"% repr(player.genmove(s, 'w').move))

    test_bot()
