from __future__ import print_function

import tempfile
import subprocess
import os
import logging
import array
import numpy as np
import time

from players import DistributionBot, DistWrappingMaxPlayer

import cubes
from state import State
import rank

class DeepCL_IO(object):
    def __init__(self,
                 deepclexec_path,
                 options={
                     "dataset" : "kgsgo", # needed for normalization
                     # "weightsFile": "weights.dat", # default value
                     # see 'deepclexec -h' for other options
                     # CAVEEAT: normalization has to be set up the same
                     # as when the CNN was trained
                     },
                 shape=(7, 19, 19) # it is a bit ugly, but DeepCL reads
                                   # the shape info before it opens the output
                                   # file, so we cannot wait until first run
                                   # to read the info from the cube
                                   # (or we could postpone the initialization
                                   # until first call, but this is ugly)
                 ):
        # DeepCL works with 4 byte floats, so we need to ensure we have
        # the same size, if this fails, we could probably reimplement it
        # using struct module
        self.itemsize = 4
        a = array.array('f')
        assert a.itemsize == self.itemsize

        self.deepclexec_path = deepclexec_path

        for res_opt in ['outputfile', 'batchsize']:
            if res_opt in options:
                logging.warn("DeepCL_IO: '%s' option is reserved, overriding."%res_opt)

        options['batchsize'] = 1

        # first create the named pipes for IO
        self.pipe_to, self.pipe_from = None,  None

        self.tempdir = tempfile.mkdtemp()

        self.pipe_fn_from = os.path.join(self.tempdir, "PIPE_from")

        options['outputfile'] = self.pipe_fn_from

        os.mkfifo(self.pipe_fn_from)

        self.p = subprocess.Popen([deepclexec_path] + [ "%s=%s"%(k, v) for k, v in options.iteritems() ],
                                  stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT)

        logging.debug("Waiting 3sec for the deepclexec to start up properly..")
        time.sleep(3)
        # if the process is dead already,
        # we cannot proceed, we would hang on the first pipe (below)
        if self.p.poll() != None:
            logging.debug("deepclexec died unexpectedly")
            self.gather_sub_logs()
            raise RuntimeError("deepclexec died unexpectedly")

        # us -> them
        self.pipe_to = self.p.stdin
        # write header
        # cube shape = 7 x 19 x 19
        shapea = np.array(shape, dtype='i4')
        shapea.tofile(self.pipe_to)
        self.pipe_to.flush()

        # them -> us
        # this might seem like a too verbose piece of code, but
        # unfortunately, open() hangs if the other side is not opened,
        # so it is good to see this in log in the case...
        try:
            logging.debug("Setting up pipe: "+ self.pipe_fn_from)
            self.pipe_from = open(self.pipe_fn_from, 'rb')
            logging.debug("Pipes set up.")
        except KeyboardInterrupt:
            self.gather_sub_logs()
            raise

    def gather_sub_logs(self):
        logging.debug("Gathering subprocess logs.")
        stdout, _stderr =  self.p.communicate()
        logging.debug("Ended with returncode %d."%self.p.returncode)
        # see def of self.p, 2>&1
        logging.debug("stdout + stderr:\n"+str(stdout) +"\n")

    def close_pipes(self):
        if self.pipe_to !=  None:
            self.pipe_to.close()

        if self.pipe_from !=  None:
            self.pipe_from.close()

    def close(self):
        #self.close_pipes()
        self.gather_sub_logs()
        #self.p.terminate()

        os.unlink(self.pipe_fn_from)
        os.rmdir(self.tempdir)

    def write_cube(self, cube):
        cube.tofile(self.pipe_to)
        self.pipe_to.flush()

    def read_response(self, side):
        return np.fromfile(self.pipe_from, dtype="float32", count=side*side)

    def interact(self, cube, side):
        self.write_cube(cube)
        return self.read_response(side)


class DeepCLDistBot(DistributionBot):
    def __init__(self, deepcl_io):
        super(DeepCLDistBot,  self).__init__()
        self.deepcl_io = deepcl_io

    def gen_probdist_raw(self, state, player):
        cube = cubes.get_cube_deepcl(State(state.board,
                                           state.ko_point,
                                           # history and ranks are not used by
                                           # deepcl cubes
                                           [], BrWr(None, None)),
                                     player)

        try:
            logging.debug("Sending data, cube.shape = %s, %d B"%(cube.shape,
                                                                 self.deepcl_io.itemsize * reduce(lambda a, b:a*b, cube.shape)))
            response = self.deepcl_io.interact(cube, side=state.board.side)
        except:
            #self.deepcl_io.close_pipes()
            self.deepcl_io.gather_sub_logs()
            raise

        logging.debug("Got response of size %d B"%(self.deepcl_io.itemsize * len(response)))

        return response.reshape((state.board.side, state.board.side))

    def close(self):
        self.deepcl_io.close()


if __name__ == "__main__":
    def test_bot():
        import gomill

        logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                            level=logging.DEBUG)

        DCL_PATH = '/home/jm/prj/DeepCL/'
        deepcl_io = bot_deepcl.DeepCL_IO(os.path.join(DCL_PATH, 'build/predict'), options={
            'weightsfile': os.path.join(DCL_PATH, "build/weights.dat"),
            'outputformat': 'binary',
                })

        player = DistWrappingMaxPlayer(DeepCLDistBot(deepcl_io))

        class State:
            pass
        s = State()

        b = gomill.boards.Board(19)
        s.board = b
        s.ko_point = None
        logging.debug("bot: %s"% repr(player.genmove(s, 'w').move))

        #player.handle_quit([])

    test_bot()
