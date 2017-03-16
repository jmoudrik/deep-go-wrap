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
from gomill.gtp_states import History_move

from deepgo import cubes, state, rank

"""
This reads sgf's from stdin, processes them in a parallel manner to extract
pairs (cube_encoding_position, move_to_play) and writes the data into a file.

Some comments about speed:

Most time in workers is currently spent in routines for analysing the goban
in the cubes submodule, where we analyse each position independently of
the previous ones, while we could build strings/liberties data structures
incrementaly and thus save resources.

The workers do however scale up linearly with number of cores. What does not
and what is the actual bottleneck on multicore machine (with slower & bigger
cubes, such as the tian_zhu_2015 cube) is the serial HDF file io and compression
in the master process. Truly parallel implementation using MPI is planned.

Currently, you can easily process 200 000 games in under a 24 hours on 4-core
commodity laptop. The dataset is created (almost) only once and you will
probably be spending much more time training the CNN anyway.
"""


def flatten(list_of_lists):
    return chain.from_iterable(list_of_lists)

def init_subprocess(plane, label, allowed_boardsizes, allowed_ranks):
    global get_cube, get_label, board_filter, ranks_filter
    get_cube = cubes.reg_cube[plane]
    get_label = cubes.reg_label[label]
    board_filter = lambda board : board.side in allowed_boardsizes

    def filter_one_rank(rank):
        if allowed_ranks is None:
            return True
        if not rank:
            return None in allowed_ranks
        return rank.key() in allowed_ranks

    def ranks_filter(brwr):
        return all(map(filter_one_rank, brwr))

def get_rank(root_node, key):
    try:
        prop = root_node.get(key)
    except:
        return None

    return rank.Rank.from_string(prop, True)

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
        logging.info("Skipping game '%s': boardsize not allowed"%(sgf_fn))
        return None

    root = game.get_root()
    ranks = rank.BrWr(get_rank(root, 'BR'),
                      get_rank(root, 'WR'))

    if not ranks_filter(ranks):
        logging.info("Skipping game '%s': rank not allowed"%(sgf_fn))
        return None

    Xs = []
    ys = []

    ko_move = None
    history = []
    for num, (player, move) in enumerate(moves):
        # pass
        if not move:
            break

        try:
            # encode current position
            s = state.State(board, ko_move, history, moves[num:len(moves)], ranks)
            x = get_cube(s, player)
            # get y data from future moves
            # (usually only first element will be taken in account)
            y = get_label(s, player)
        except cubes.SkipGame as e:
            logging.info("Skipping game '%s': %s"%(sgf_fn, str(e)))
            return None
        except Exception as e:
            logging.exception("Error encoding '%s' - move %d"%(sgf_fn, num + 1))
            # TODO Should we use the data we have already?
            return None

        # None skips
        if x is not None and y is not None:
            Xs.append(x)
            ys.append(y)

        row, col = move
        try:
            ko_move = board.play(row, col, player)
        except Exception as e:
            logging.warn("Error re-playing '%s' - move %d : '%s'"%(sgf_fn, num + 1, str(e)))
            # this basically means that the game has illegal moves
            # lets skip it altogether in case it is garbled
            return None
        history.append(History_move(player, move))

    return Xs, ys

def parse_rank_specification(s):
    """
    Parses info about rank specification, used to filter games by player's ranks.
    Returns None (all ranks allowed),
    or a set of possible values
    (None as a possible value in the set means that we should include games without rank info)


    # returns None, all ranks possible
    parse_rank_specification('')

    # returns set([1, 2, 3, None])), 1, 2, 3 allowed, as well as missing rank info
    parse_rank_specification('1..3,')

    # returns set([None])), only games WITHOUT rank info are allowed
    parse_rank_specification(',')

    See test for more examples.
    """
    if not s:
        return None

    ret = []
    s = s.replace(' ','')
    categories = s.split(',')

    for cat in categories:
        cs = cat.split('..')
        try:
            if len(cs) == 1:
                if not cs[0]:
                    ret.append(None)
                else:
                    ret.append(int(cs[0]))
            elif len(cs) == 2:
                fr, to = map(int,cs)
                if to < fr:
                    raise RuntimeError('Empty range %s'%(cat))

                ret.extend(range(fr, to+1))
            else:
                raise ValueError()

        except ValueError:
            raise RuntimeError('Could not parse rank info on token "%s"'%(cat))

    return set(ret)

class RankSpecAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
             raise ValueError("nargs not allowed")
        super(RankSpecAction, self).__init__(option_strings, dest, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, parse_rank_specification(values))


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
    parser.add_argument('-x', '--x-name',  dest='xname',
                        help='HDF5 dataset name to store the xs to',
                        default='xs')
    parser.add_argument('-y', '--y-name', dest='yname',
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
    parser.add_argument('-r', '--rank', dest='rankspec', action=RankSpecAction,
                        help='Specify rank to be limited. 1kyu=1, 30kyu=30, 1dan=0,'
                             ' 10dan=-9. Example values "1,2,3", "1..30", "-10..30",'
                             ' etc. Empty string marks no limit on rank.',
                        default=parse_rank_specification(''))
    parser.add_argument('--flatten', dest='flatten', action='store_true',
                        help='Flatten out the examples. (19, 19, 4) shape becomes ( 19 * 19 * 4,)', default=False)
    parser.add_argument('--shrink-units', dest='shrink_units', action='store_true',
                        help='Shrinks unit dimension label (or, unlikely, feature) arrays.'
                             ' Only if the unit dimension is the only one in the example,'
                             ' so (19,19,1) is not shrinked, but (1,) is.', default=False)
    parser.add_argument('--dtype', dest='dtype',
                        help='convert dtype of stored data to given numpy dtype (instead the default value defined by plane/label)', default=None)
    parser.add_argument('--compression', dest='compression',
                        help='Possible values: "lzf", "gzip10", "gzip9", ...', default='lzf')
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

    initargs=(args.plane, args.label, (args.boardsize, ), args.rankspec)
    p = multiprocessing.Pool(args.proc, initializer=init_subprocess, initargs=initargs)

    ## INIT shapes and transformations
    # the basic pathway is:
    # imap job returns two lists [x1, x2, x3, ..], [y1, y2, y3, ..]
    # of numpy arrays which we want to transform to be able to store in a dataset
    # in a proper format

    # first determine example shapes
    b = gomill.boards.Board(args.boardsize)
    init_subprocess(*initargs)
    s = state.State(b, None, [], [('b',(3,3))], rank.BrWr(rank.Rank.from_key(1), # 1k
                                               rank.Rank.from_key(2)  # 2k
                                               ))
    sample_x = get_cube(s, 'b')
    sample_y = get_label(s, 'b')

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

    ## compression
    compression_kwargs = {}
    if args.compression == 'lzf':
        compression_kwargs['compression'] = 'lzf'
    elif args.compression.startswith('gzip'):
        compression_kwargs['compression'] = 'gzip'
        level = int(args.compression[4:])
        assert 0<=level<=10
        compression_kwargs['compression_opts'] = level
    else:
        raise RuntimeError("Invalid compression arg.")

    ## INIT dataset
    with h5py.File(args.filename) as f:
        logging.debug("what: raw -> in dataset")
        logging.debug("x.shape: %s -> %s"%(repr(sample_x.shape), repr(dshape_x) if dshape_x else 'flat'))
        logging.debug("x.dtype: %s -> %s"%(sample_x.dtype, dtype_x))
        logging.debug("y.shape: %s -> %s"%(repr(sample_y.shape), repr(dshape_y) if dshape_y else 'flat'))
        logging.debug("y.dtype: %s -> %s"%(sample_y.dtype, dtype_y))

        try:
            kwargsx = {
                # infinite number of examples
                'maxshape' :(None,) + dshape_x,
                'dtype' : dtype_x,
            }
            kwargsx.update(compression_kwargs)

            dset_x = f.create_dataset(args.xname,
                                      (0,) + dshape_x,
                                      **kwargsx)
            kwargsy = {
                # infinite number of examples
                'maxshape' :(None,) + dshape_y,
                'dtype' : dtype_y,
            }
            kwargsy.update(compression_kwargs)

            dset_y = f.create_dataset(args.yname,
                                      (0,) + dshape_y,
                                      **kwargsy)
        except Exception as e:
            logging.error("Cannot create dataset. File exists? (%s)"%(str(e)))
            sys.exit(1)


        dset_x.attrs['name'] = args.plane
        dset_y.attrs['name'] = args.label
        dset_x.attrs['boardsize'] = args.boardsize
        dset_y.attrs['boardsize'] = args.boardsize
        dset_x.attrs['original_dtype'] = repr(sample_x.dtype)
        dset_y.attrs['original_dtype'] = repr(sample_y.dtype)
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


