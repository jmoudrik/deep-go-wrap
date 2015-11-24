#!/usr/bin/env python

import sys
import logging
import multiprocessing
from itertools import imap, chain, islice
import argparse
import numpy as np

import h5py
import gomill
import gomill.sgf, gomill.sgf_moves

from deepgo import cubes

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

        logging.info("Processing '%s'"%sgf_fn)
        board, moves = gomill.sgf_moves.get_setup_and_moves(game)
    except Exception as e:
        logging.warn("Error processing '%s': %s"%(sgf_fn, str(e)))
        return None

    if not board_filter(board) or not moves:
        return None

    Xs = []
    ys = []

    ko_move = None
    for num, (player, move) in enumerate(moves):
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
        try:
            ko_move = board.play(row, col, player)
        except Exception as e:
            logging.warn("Error processing '%s' - move %d : '%s'"%(sgf_fn, num + 1, str(e)))
            return None

    return Xs, ys

def parse_args():
    parser = argparse.ArgumentParser(
                description='Processes sgf to create datasets for teaching Deep'
                            ' Neural Networks to play the game of Go.'
                            ' Each sgf file is read from STDIN, analysed and an'
                            ' (X, y) pair is created from each position, where'
                            ' X is the cube encoding position and y the desired'
                            ' move. The results are written to HDF5 file.')
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
    parser.add_argument('--flatten', dest='flatten', action='store_true',
                        help='Flatten out the examples. (19, 19, 4) shape becomes ( 19 * 19 * 4,)', default=False)
    parser.add_argument('--shrink-units', dest='shrink_units', action='store_true',
                        help='Shrinks unit dimension label (or, unlikely, feature) arrays.'
                             ' Only if the unit dimension is the only one in the example,'
                             ' so (19,19,1) is not shrinked, but (1,) is.', default=False)
    parser.add_argument('--dtype', dest='dtype',
                        help='convert dtype of stored data to given numpy dtype (instead the default value defined by plane/label)', default=None)
    parser.add_argument('--proc', type=int,
                        default=multiprocessing.cpu_count(),
                        help='specify number of processes for parallelization')

    return parser.parse_args()


def batched_imap(function, input_iterator, batch_size=100, imap=imap):
    """
        Runs `function` using `imap` on batches of `batch_size`
        taken from `input_iterator`.
        yield results.

        Only runs next imap when all previous results have been consumed.
        This is useful if workers in the imap pool are faster than consumer
        of the results, because the results might use up a lot of memory.
    """
    def next_batch():
        # list is necessary s.t. we can test for emptiness
        return list(islice(input_iterator, batch_size))

    batch = next_batch()
    while batch:
        logging.debug('Starting next batch.')
        for res in imap(function, batch):
            yield res
        batch = next_batch()


def main():
    ## ARGS
    args = parse_args()

    ## INIT LOGGING
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                        level=logging.DEBUG) # if not args.quiet else logging.WARN)

    logging.info("args: %s"%args)

    ## INIT pool of workers

    initargs=(args.plane, args.label, (args.boardsize, ))
    p = multiprocessing.Pool(args.proc, initializer=init_subprocess, initargs=initargs)

    ## INIT shapes and transformations
    # the basic pathway is:
    # imap job returns two lists [x1, x2, x3, ..], [y1, y2, y3, ..]
    # of numpy arrays which we want to transform to be able to store in a dataset
    # in a proper format

    # first determine example shapes
    b = gomill.boards.Board(args.boardsize)
    init_subprocess(*initargs)
    sample_x = get_cube(b, None, 'b')
    sample_y = get_label((0, 0), args.boardsize)

    # shape in dataset
    dshape_x = sample_x.shape
    dshape_y = sample_y.shape

    # transformation of the returned lists
    mapxs = lambda xs : xs
    mapys = lambda ys : ys

    ## shrink unit dimension
    # one dimensional values can be stored flattened
    # s.t.
    # 1000 examples of dimensions 1 have shape (1000,)
    # instead of (1000, 1)
    # this is probably the case only for the labels
    # but support xs anyways
    if args.shrink_units and sample_x.shape == (1, ):
        mapxs = lambda xs : np.ndarray.flatten(np.array(xs))
        dshape_x = tuple()

    if args.shrink_units and sample_y.shape == (1, ):
        mapys = lambda ys : np.ndarray.flatten(np.array(ys))
        dshape_y = tuple()

    transform_example_x = lambda x : x
    transform_example_y = lambda y : y

    ## flatten
    # do not flatten units
    if args.flatten and dshape_x:
        transform_example_x = np.ndarray.flatten
        dshape_x = (reduce((lambda x,y : x*y), dshape_x), )

    if args.flatten and dshape_y:
        transform_example_y = np.ndarray.flatten
        dshape_y = (reduce((lambda x,y : x*y), dshape_y), )

    ## dtype
    dtype_x = sample_x.dtype
    dtype_y = sample_y.dtype

    recast_dtype = lambda a : a
    if args.dtype:
        recast_dtype = lambda a : np.array(a, dtype=args.dtype)
        dtype_x = args.dtype
        dtype_y = args.dtype


    ## INIT dataset
    with h5py.File(args.filename) as f:
        logging.debug("what: raw -> in dataset")
        logging.debug("x.shape: %s -> %s"%(repr(sample_x.shape), repr(dshape_x) if dshape_x else 'flat'))
        logging.debug("x.dtype: %s -> %s"%(sample_x.dtype, dtype_x))
        logging.debug("y.shape: %s -> %s"%(repr(sample_y.shape), repr(dshape_y) if dshape_y else 'flat'))
        logging.debug("y.dtype: %s -> %s"%(sample_y.dtype, dtype_y))

        dset_x = f.create_dataset(args.xname,
                                  (0,) + dshape_x,
                                  # infinite number of examples
                                  maxshape=(None,) + dshape_x,
                                  dtype=dtype_x,
                                  # we will have a lot of zeros in the data
                                  compression='gzip', compression_opts=9)

        dset_y = f.create_dataset(args.yname,
                                  (0,) + dshape_y,
                                  maxshape=(None,) + dshape_y,
                                  dtype=dtype_y,
                                  compression='gzip', compression_opts=9)

        dset_x.attrs['name'] = args.plane
        dset_y.attrs['name'] = args.label
        dset_x.attrs['boardsize'] = args.boardsize
        dset_y.attrs['boardsize'] = args.boardsize
        dset_x.attrs['original_dtype'] = repr(sample_x.dtype)
        dset_y.attrs['original_dtype'] = repr(sample_x.dtype)
        dset_x.attrs['original_example_shape'] = repr(sample_x.shape)
        dset_y.attrs['original_example_shape'] = repr(sample_y.shape)

        ## map the job

        if args.proc > 1:
            def job_imap(*args):
                return p.imap_unordered(*args)
        else:
            # do not use pool if only one proc
            init_subprocess(*initargs)
            def job_imap(*args):
                return imap(*args)

        it = batched_imap(process_game, sys.stdin, batch_size=1000, imap=job_imap)

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
                logging.info("Storing %d examples."%add)
                dset_x.resize((size+add,) + dshape_x)
                dset_y.resize((size+add,) + dshape_y)

                dset_x[-add:] = mapxs([transform_example_x(recast_dtype(x)) for x in xs])
                dset_y[-add:] = mapys([transform_example_y(recast_dtype(y)) for y in ys])

                size += add

        logging.info("Finished.")
        for dset in [dset_x, dset_y]:
            logging.info("Dataset '%s': shape=%s, size=%s, dtype=%s"%(dset.name,
                                                                       repr(dset.shape),
                                                                       repr(dset.size),
                                                                       repr(dset.dtype)))

if __name__ == "__main__":
    main()


