#!/usr/bin/env python

import logging
import argparse
import numpy as np

import h5py

def parse_args():
    parser = argparse.ArgumentParser(
                description='Converts the HDF5 dataset to the binary'
                            ' format v2 compatible with DeepCL.'
                            ' The v2 format specification is available at:'
                            ' https://github.com/hughperkins/kgsgo-dataset-preprocessor'
                            ' The HDF5 dataset must be created using'
                            ' these two options:'
                            ' "-p clark_storkey_2014_packed -l simple_label"'
                            ' or this will fail.')
    parser.add_argument('filename_in', metavar='FILENAME_IN',
                        help='HDF5 filename to read the dataset to')
    parser.add_argument('filename_out', metavar='FILENAME_OUT',
                        help='deepcl v2 filename to store result to')
    parser.add_argument('--x-name',  dest='xname', 
                        help='HDF5 dataset name to read the xs from',
                        default='xs')
    parser.add_argument('--y-name', dest='yname', 
                        help='HDF5 dataset name to read the ys from', 
                        default='ys')
    

    return parser.parse_args()


def main():
    ## ARGS
    args = parse_args()
    
    ## INIT LOGGING
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                        level=logging.DEBUG) # if not args.quiet else logging.WARN)
    
    logging.info("args: %s"%args)
    
    ## INIT dataset
    with h5py.File(args.filename_in) as hdf:
        dset_x = hdf[args.xname]
        dset_y = hdf[args.yname]
        
        if dset_x.attrs['name'] != 'clark_storkey_2014_packed':
            logging.error("The input dataset must have planes as specified by opt: -p clark_storkey_2014_packed")
        if dset_y.attrs['name'] != 'simple_label':
            logging.error("The input dataset must have label as specified by opt: -l simple_label")
            
        assert dset_x.shape[0] == dset_y.shape[0]
        num_examples = dset_x.shape[0]
        
        with open(args.filename_out, 'w') as fout:
            logging.info("Starting the conversion to v2")
            
            header = '-'.join(["mlv2",
                               "n=%d" % num_examples, 
                               "numplanes=7", 
                               "imagewidth=%d" % dset_x.attrs['boardsize'], 
                               "imageheight=%d"% dset_x.attrs['boardsize'], 
                               "datatype=int", 
                               "bpp=1\0\n"])
            fout.write(header)
            # the header is padded
            fout.write( chr(0) * (1024 - len(header)))
            
            for i in xrange(num_examples):
                if i and i % 10000 == 0:
                    logging.info("Processed %d / %d = %.1f%%"%(i,
                                                               num_examples,
                                                               100.0*i/num_examples))
                data, label = dset_x[i], dset_y[i]
                
                # each example is prefixed by 'GO' string
                fout.write('GO')
                
                # then label
                label_high, label_low = label // 256, label % 256
                fout.write(chr(label_low))
                fout.write(chr(label_high))
                fout.write(chr(0) * 2)
                
                # then the planes
                # clark_storkey_2014_packed, has just the correct representation
                data.tofile(fout)
                
            # finaly, mark the end
            fout.write('END')
                
            logging.info("Finished processing %d examples."%(num_examples))
            
if __name__ == "__main__":
    main()

        
