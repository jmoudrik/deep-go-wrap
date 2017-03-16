from contextlib import contextmanager
from itertools import count
from unittest import TestCase
from functools import reduce
import os
import numpy as np

import h5py

import hdf_utils


@contextmanager
def removing_files(filename_gen):
    files = []

    def generator():
        for name in filename_gen:
            files.append(name)
            yield name

    try:
        yield generator

    finally:
        for filename in files:
            if os.path.exists(filename):
                os.unlink(filename)


def counting_namefactory(prefix="tempfile", suffix=".tmp"):
    for i in count():
        fn = "%s%d%s" % (prefix, i, suffix)
        if not os.path.exists(fn):
            yield fn


def mult(iterable):
    return reduce((lambda x, y: x * y), iterable)


def make_test_dset(fname):
    fout = h5py.File(fname, 'w')

    length = 100
    dshape_x = (3, 5)
    dshape_y = (2, 7)
    dset_x = fout.create_dataset('xs',
                                 (length,) + dshape_x,
                                 dtype='uint')

    dset_y = fout.create_dataset('ys',
                                 (length,) + dshape_y,
                                 dtype='uint')

    dset_x[:] = np.arange(mult(dset_x.shape)).reshape(dset_x.shape)
    dset_y[:] = np.arange(mult(dset_y.shape)).reshape(dset_y.shape)

    return dset_x, dset_y, fout


class Test(TestCase):
    def test_split_merge(self):
        with removing_files(counting_namefactory('tempfile_', '.%d.tmp' % os.getpid())) as nameg_factory:
            name_it = nameg_factory()

            name = next(name_it)
            dx, dy, fout = make_test_dset(name)
            ax, ay = np.array(dx), np.array(dy)
            fout.close()

            tosplit = [hdf_utils.HdfLoc(next(name_it), 'xs', 'ys') for _ in range(20)]
            hdf_utils.split(hdf_utils.HdfLoc(name, 'xs', 'ys'),
                            [hdf_utils.SplitTo(dest, 5 * i if i < len(tosplit) - 1 else -1) for i, dest in
                             enumerate(tosplit)],
                            blocksize=11)

            namem = next(name_it)
            hdf_utils.merge(hdf_utils.HdfLoc(namem, 'xs', 'ys'),
                            tosplit,
                            blocksize=13)

            merged = h5py.File(namem, 'r')
            mx = np.array(merged['xs'][:])
            my = np.array(merged['ys'][:])

            assert mx.shape == ax.shape
            assert my.shape == ay.shape
            assert (mx == ax).all()
            assert (my == ay).all()


if __name__ == '__main__':
    import unittest

    unittest.main()
