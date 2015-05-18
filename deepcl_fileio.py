from __future__ import print_function

import tempfile
import subprocess
import os
import logging
import array
import numpy as np
import time

from players import DistributionBot, DistWrappingMaxPlayer
import gomill

import cubes

class DeepCL_FileIO(object):
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

        for res_opt in ['inputfile', 'outputfile']:
            if res_opt in options:
                logging.warn("DeepCL_IO: '%s' option is reserved, overriding."%res_opt)

        # first create the named pipes for IO
        self.tempdir = tempfile.mkdtemp()

        self.fn_to = os.path.join(self.tempdir, "FILE_to")
        self.fn_from = os.path.join(self.tempdir, "FILE_from")

        options['inputfile'] = self.fn_to
        options['outputfile'] = self.fn_from
        
        self.deepclexec_path = deepclexec_path
        self.options = options
        
    def write_cube(self, cube):
        cube.tofile(self.pipe_to)
        self.pipe_to.flush()

    def read_response(self, side):
        return np.fromfile(self.pipe_from, dtype="float32", count=side*side)
    
    def interact(self, cube, side):
        ## write input
        with open(self.fn_to, 'wb') as fout:
            cube.tofile(fout)
            
        ## compute response
        self.p = subprocess.Popen([self.deepclexec_path] + [ "%s=%s"%(k, v) for k, v in self.options.iteritems() ],
                                  stdin=None,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT)
        
        logging.debug("Gathering subprocess logs.")
        stdout, stderr =  self.p.communicate()
        logging.debug("stdout:\n"+str(stdout) +"\n")
        #logging.debug("stderr:\n"+str(stderr) +"\n")

        ## read output
        with open(self.fn_from, 'rb') as fin:
            response = np.fromfile(fin, dtype="float32", count=side*side)
            
        return response
            
    def close(self):
        if os.path.exists(self.fn_to):
            os.unlink(self.fn_to)
        if os.path.exists(self.fn_from):
            os.unlink(self.fn_from)
        os.rmdir(self.tempdir)
