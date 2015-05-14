# deep-go-wrap
Toolkit designed to ease development of your Deep Neural Network models
for the game of Go.  *Main feature* is the *GTP wrapper* which makes it
easy to turn raw probability distribution Convolutional Neural Network
models (or whatever) into full-featured GTP players.

Another feature is a *data preprocessor* for creating arbitrary *Go
datasets*, mainly to be used with CNNs.  We store these in a fairly
universal HDF5 format (supports compression transparently, large sizes,
has wrappers for a lot of languages). Extracting different planes from
positions, making labels, you name it! :-)

GTP wrapper
-----------
 * interface for plugging in various Deep Network architectures
    * so far only DeepCL nets, easy to extend
 * I/O handling, data planes extraction
 * full GTP support using gomill library
 * pass implementation, by using GnuGo as an pass-oracle
 * TODO:
     * move correctness checking
     * **any other ideas?**

Dataset Processing
------------------
  * different planes support, easy to extend
     * as in [Clark and Storkey 2014](http://arxiv.org/abs/1412.3409)
     * planes from DeepCL
     * others to come
  * parallel processing

#### A naive bash example how to make a dataset
```bash
DATA_DIR="/path/to/dir/containing/your/games"

# make pseudorandomly shuffled list of all games found
find "$DATA_DIR" -name '*.sgf' | sort -R > filelist

# this creates the dataset, with 1 bit per plane (7 of them)
# for each goban point. The dataset is transparently gzip
# compressed by hdf5, so the size is managable.
cat filelist | ./process_sgf.py -p clark_storkey_2014_packed dataset.hdf5
```

Requirements
------------
 * [gomill library](https://github.com/mattheww/gomill)
 * [hdf5 library](http://www.h5py.org/) if you wish to make Datasets 
