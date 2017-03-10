import numpy as np

from utils import border_mark, l2_distance, gridcular_distance, distances_from_pt, sq_distance

CACHE = {}
def cached(function):
    def f2(*args):
        global CACHE
        key = function.__name__ + repr(args)
        if key in CACHE:
            return CACHE[key]

        ret = function(*args)
        CACHE[key] = ret
        return ret
    return f2

@cached
def get_border_mark(boardsize):
    return border_mark(boardsize)

@cached
def get_l2_from_center(boardsize):
    assert boardsize % 2 == 1
    center = boardsize // 2
    return distances_from_pt(l2_distance, (center, center), boardsize)

@cached
def get_sqd_from_center(boardsize):
    """
    plane `position mask` from
    Yuandong Tian, Yan Zhu, 2015
    Better Computer Go Player with Neural Network and Long-term Prediction
    (arXiv:1511.06410)
    """
    assert boardsize % 2 == 1
    center = boardsize // 2
    return distances_from_pt(sq_distance, (center, center), boardsize)

@cached
def get_gridcular_from_center(boardsize):
    assert boardsize % 2 == 1
    center = boardsize // 2
    return distances_from_pt(gridcular_distance, (center, center), boardsize)

@cached
def get_exp_gridcular_from_center(boardsize, exp):
    return np.exp(exp * get_gridcular_from_center(boardsize))

@cached
def get_exp_sqd_from_center(boardsize, exp):
    return np.exp(exp * get_sqd_from_center(boardsize))

if __name__ == "__main__":
    import time
    for a in xrange(4000000):
        pass

    t0 = time.time()
    a = get_exp_gridcular_from_center(19, -0.5)
    t1 = time.time()
    b = get_exp_gridcular_from_center(19, -0.5)
    t2 = time.time()

    print (t1 - t0) / (t2 - t1)
    a = get_gridcular_from_center(19)
    #a = get_exp_gridcular_from_center(19, -0.5)
    print a
