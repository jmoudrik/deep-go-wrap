#!/usr/bin/env python
from __future__ import print_function

import h5py
from collections import namedtuple

HdfLoc = namedtuple('HdfDestination', ['filename', 'xkey', 'ykey'])
SplitTo = namedtuple('SplitTo', ['dest', 'at'])


def print_stats(files):
    def print_one_obj(key, obj):
        print(obj)
        if isinstance(obj, h5py.Dataset):
            for k, v in sorted(obj.attrs.items()):
                print("%25s\t%s" % (k, v))
        elif isinstance(obj, h5py.Group):
            for k, v in sorted(obj.items()):
                print("%25s\t%s" % (k, v))
        else:
            assert False

    for f in files:
        hf = h5py.File(f, 'r')
        print('=' * 40)
        print(f)
        print('=' * 40)
        for key, obj in hf.items():
            print_one_obj(key, obj)


def copy_attrs(source, dest):
    for k, v in source.attrs.items():
        dest.attrs[k] = v


def append_data(source, dest, start, end):
    data = source[start:end]
    assert source.shape[1:] == dest.shape[1:]

    add = data.shape[0]
    dest.resize((dest.shape[0] + add,) + dest.shape[1:])
    dest[-add:] = data

    return add


def open_copy_output(dest, orig_x, orig_y):
    fout = h5py.File(dest.filename, 'a')
    dset_x = fout.create_dataset(dest.xkey,
                                 (0,) + orig_x.shape[1:],
                                 maxshape=(None,) + orig_x.shape[1:],
                                 dtype=orig_x.dtype,
                                 # we will have a lot of zeros in the data
                                 compression='lzf')

    dset_y = fout.create_dataset(dest.ykey,
                                 (0,) + orig_y.shape[1:],
                                 maxshape=(None,) + orig_y.shape[1:],
                                 dtype=orig_y.dtype,
                                 # we will have a lot of zeros in the data
                                 compression='lzf')

    copy_attrs(orig_x, dset_x)
    copy_attrs(orig_y, dset_y)

    return dset_x, dset_y, fout


def split(source, splits, blocksize=100000):
    fin = h5py.File(source.filename, 'r')
    dshape_x = fin[source.xkey].shape
    dshape_y = fin[source.ykey].shape

    lx, ly = dshape_x[0], dshape_y[0]
    assert lx == ly

    start = 0
    for current_split in splits:
        dset_x, dset_y, fout = open_copy_output(current_split.dest, fin[source.xkey], fin[source.ykey])
        # print split.dest.filename

        finsize = min(dshape_x[0], current_split.at if current_split.at >= 0 else dshape_x[0])
        while start < finsize:
            end = min(start + blocksize, finsize)
            # print start, end, finsize
            added_x = append_data(fin[source.xkey], dset_x, start, end)
            added_y = append_data(fin[source.ykey], dset_y, start, end)
            # TODO we should really check for exceptions in append data
            # and rollback the block in case of failure
            assert added_x == added_y

            start += blocksize

        fout.close()

        # we have walked through the whole input file
        if start >= dshape_x[0]:
            break

        # we just finished last split
        # so continue from where we left off
        start = finsize
        assert start < dshape_x[0]


def merge(target, sources, blocksize=100000):
    assert len(sources) >= 2

    dset_x, dset_y, fout = None, None, None
    for source in sources:
        fin = h5py.File(source.filename, 'r')
        dshape_x = fin[source.xkey].shape
        dshape_y = fin[source.ykey].shape

        lx, ly = dshape_x[0], dshape_y[0]
        assert lx == ly

        if fout is None:
            dset_x, dset_y, fout = open_copy_output(target, fin[source.xkey], fin[source.ykey])

        # we append the source onto fout
        start, finsize = 0, lx
        while start < finsize:
            end = min(start + blocksize, finsize)
            # print start, end, finsize

            added_x = append_data(fin[source.xkey], dset_x, start, end)
            added_y = append_data(fin[source.ykey], dset_y, start, end)
            assert added_x == added_y

            start += blocksize

    if fout is not None:
        fout.close()



def usage():
    print("""%s <command> [args]
Command can be one of:
    i, identify     print info of groups in the hdf files [args]
    s, split        args:
    """)


"""
def main():
    ## ARGS
    args = parse_args()
"""

if __name__ == "__main__":
    import sys

    argc, argv = len(sys.argv), sys.argv
    if argc > 2 and argv[1].lower() in ['i', 'identify']:
        print_stats(argv[2:])

    if argc >= 2 and argv[1].lower() in ['s', 'split']:
        split(HdfLoc('gokifu_1_20000.hdf', 'xs', 'ys'),
              #[SplitTo(HdfLoc('gokifu_validation.hdf', 'xs', 'ys'), 200000),
              [SplitTo(HdfLoc('output_0.hdf', 'foo/xs1', 'foo/ys1'), 10),
               SplitTo(HdfLoc('output_0.hdf', 'foo/xs2', 'foo/ys2'), 17),
               ],
              blocksize=4)

    if argc >= 5 and argv[1].lower() in ['m', 'merge']:
        target, rest = argv[2], argv[3:]
        merge(HdfLoc(target, 'xs', 'ys'),
              [HdfLoc(t, 'xs', 'ys') for t in rest])
