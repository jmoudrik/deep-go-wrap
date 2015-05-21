#!/usr/bin/env python

import sys
import logging
import multiprocessing
from itertools import imap, chain
import argparse
import numpy as np

import h5py
import gomill
import gomill.sgf, gomill.sgf_moves

import cubes

"""
This reads sgf's from stdin, processes them in a parallel manner to extract
pairs (cube_encoding_position, move_to_play) and writes the data into a file.

The major bottleneck is currently the inneficiency of analysing the goban
in the cubes submodule, where we analyse each position independently of
the previous ones, while we could build strings/liberties data structures
incrementaly and thus save resources.

But I think this is not worth the effort at the moment; you can easily
process 200 000 games in under a 24 hours on 4-core commodity laptop. The
dataset is created (almost) only once and you will probably be spending much
more time training the CNN anyway.
"""


def flatten(list_of_lists):
    return chain.from_iterable(list_of_lists)

def init_subprocess(plane, label, allowed_boardsizes):
    global get_cube, get_label, board_filter
    get_cube = cubes.reg_cube[plane]
    get_label = cubes.reg_label[label]
    board_filter = lambda board : board.side in allowed_boardsizes

def process_game(sgf_fn):
    sgf_fn = sgf_fn.strip()
    try :
        with open(sgf_fn, 'r') as fin:
            game = gomill.sgf.Sgf_game.from_string(fin.read())
            
        logging.debug("Processing '%s'"%sgf_fn)
        board, moves = gomill.sgf_moves.get_setup_and_moves(game)
    except Exception as e:
        logging.warn("Error processing '%s': %s"%(sgf_fn, str(e)))
        return None
        
    if not board_filter(board) or not moves:
        return None
    
    Xs = []
    ys = []
    
    ko_move = None
    for player, move in moves:
        # pass
        if not move:
            break
            
        # encode current position
        x = get_cube(board, ko_move, player)
        # target is the next move
        y = get_label(move)
        
        Xs.append(x)
        ys.append(y)
        
        row, col = move
        ko_move = board.play(row, col, player)
    
    return Xs, ys 

def parse_args():
    parser = argparse.ArgumentParser(
                description='Processes sgf to create datasets for teaching Deep'
                            ' Neural Networks to play the game of Go.'
                            ' Each sgf file is read from STDIN, analysed and an'
                            ' (X, y) pair is created from each position, where'
                            ' X is the cube encoding position and y the desired move.'
                            ' The results are written to HDF5 file.')
    parser.add_argument('filename', metavar='FILENAME',
                        help='HDF5 FILENAME to store the dataset to')
    parser.add_argument('--x-name',  dest='xname', 
                        help='HDF5 dataset name to store the xs to',
                        default='xs')
    parser.add_argument('--y-name', dest='yname', 
                        help='HDF5 dataset name to store the ys to', 
                        default='ys')
    parser.add_argument('-p', '--plane', type=str, choices=cubes.reg_cube.keys(),
                        default='clark_storkey_2014', 
                        help='specify which method should be used to create the planes')
    parser.add_argument('-l', '--label', type=str, choices=cubes.reg_label.keys(),
                        default='simple_label', 
                        help='specify which method should be used to create the labels')
    parser.add_argument('-q', '--quiet', dest='quiet', action='store_true',
                        default=False,
                        help='turn off the (stderr) debug logs')
    parser.add_argument('-s', dest='boardsize', type=int,
                        help='specify boardsize', default=19)
    parser.add_argument('--nf-units', dest='flatten_units', action='store_false', 
                        help='do not flatten out one dimensional label (or, unlikely, feature) arrays', default=True)
    parser.add_argument('--dtype', dest='dtype', 
                        help='convert dtype of stored data to given dtype (instead the default value defined by plane/label)', default=None)
    parser.add_argument('--proc', type=int,
                        default=multiprocessing.cpu_count(), 
                        help='specify number of processes for parallelization')

    return parser.parse_args()

def main():
    ## INIT ARGS
    args = parse_args()
    
    ## INIT LOGGING
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                        level=logging.DEBUG) # if not args.quiet else logging.WARN)
    
    logging.debug("args: %s"%args)
    ## INIT pool of workers
    
    initargs=(args.plane, args.label, (args.boardsize, ))
    p = multiprocessing.Pool(args.proc, initializer=init_subprocess, initargs=initargs)
    
    if args.proc > 1:
        it = p.imap_unordered(process_game, sys.stdin)
    else:
        init_subprocess(*initargs)
        it = imap(process_game, sys.stdin)
        
    ## INIT dataset
    with h5py.File(args.filename) as f:
        
        # first determine example shapes
        b = gomill.boards.Board(args.boardsize)
        init_subprocess(*initargs)
        sample_x = get_cube(b, None, 'b')
        sample_y = get_label((0, 0), args.boardsize)
        
        # one dimensional values can be stored flattened
        # s.t.
        # 1000 examples of dimensions 1 have shape (1000,)
        # instead of (1000, 1)
        # this is probably the case only for the labels 
        flat_x = args.flatten_units and sample_x.shape == (1, )
        flat_y = args.flatten_units and sample_y.shape == (1, )
        
        # shape in dataset
        dshape_x = tuple() if flat_x else sample_x.shape
        dshape_y = tuple() if flat_y else sample_y.shape
        
        logging.debug("x.shape=%s, stored %s"%(repr(sample_x.shape), 'flat' if flat_x else 'normal'))
        logging.debug("y.shape=%s, stored %s"%(repr(sample_y.shape), 'flat' if flat_y else 'normal'))
        
        dset_x = f.create_dataset(args.xname,
                                  (0,) + dshape_x,
                                  # infinite number of examples
                                  maxshape=(None,) + dshape_x, 
                                  dtype=sample_x.dtype,
                                  # we will have a lot of zeros in the data
                                  compression='gzip' )
        
        dset_y = f.create_dataset(args.yname,
                                  (0,) + dshape_y,
                                  maxshape=(None,) + dshape_y, 
                                  dtype=sample_y.dtype, 
                                  compression='gzip')
    
        size = 0
        for num, ret in enumerate(it):
            if not ret:
                continue
            
            xs, ys = ret
            assert len(xs) == len(ys)
            assert all(x.shape == sample_x.shape for x in xs)
            assert all(y.shape == sample_y.shape for y in ys)
            if xs:
                add = len(xs)
                logging.debug("Storing %d examples."%add)
                dset_x.resize((size+add,) + dshape_x)
                dset_y.resize((size+add,) + dshape_y)
                
                dset_x[-add:] = xs if not flat_x else np.ndarray.flatten(np.array(xs))
                dset_y[-add:] = ys if not flat_y else np.ndarray.flatten(np.array(ys))
                
                size += add
        
                
if __name__ == "__main__":
    main()

        