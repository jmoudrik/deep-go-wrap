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
    * so far only DeepCL nets (*working*), pylearn2 on the way, easy to extend
 * I/O handling, data planes extraction
 * full GTP support using gomill library
 * pass implementation, by using GnuGo as an pass-oracle
 * TODO:
     * move correctness checking
     * **any other ideas?**

#### DeepCL setup (under construction)
1. Train your deepcl model
    1. get data https://github.com/hughperkins/kgsgo-dataset-preprocessor
    2. get DeepCL (this is my fork of [this repo](https://github.com/hughperkins/DeepCL)
    which is sure to work, the changes aren't big and will probably be merged to the 
    Hugh's repo, so this is for the time being)
        ```$ git clone --recursive https://github.com/jmoudrik/DeepCL```
    3. train your model
    4. or, instead of the last step, just download my test small weights (precision 19.6%),
    from [here](http://j2m.cz/~jm/weights.dat) (binary file, might not work on different archs), trained with:
        ```./deepclrun numtrain=200000 dataset=kgsgoall 'netdef=32c3{z}-relu-4*(8c3{z}-relu)-128n-tanh-361n' numepochs=20 learningrate=0.0001```
2. edit path to DeepCL and model's weights file in the *deepgowrap.py* (for the time being)
    * now the deepgowrap acts as an GTP program, so you can run it:
    ```echo -e "boardsize 19\nclear_board B\nquit" | ./deepgowrap.py```

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

#### Comparison of different dataset making options
The following list summarizes file size for different options. The summary
was made from 200 random GoGoD games (39805 example pairs ~ 200 pairs per game).
Running times were basically the same (~ 1.8 sec per game on commodity laptop),
the code scales up linearly based on the number of your cores. HDF5 compresses
the dataset transparently using gzip -9. The feature cube is composed of 7 planes
used in [Clark an Storkey's 2014 paper](http://arxiv.org/pdf/1412.3409).
The ```--flatten``` option just reshapes data from ```(7,19,19)``` to ```(7*19*19,)```.

  * Fully expanded, flattened data in floats32 (e.g. ready for pylearn2)
     * 14.2kB per game: ```--dtype float32 --flatten -l expanded_label -p clark_storkey_2014```
  * Fully expanded, flattened data in uint8
     * 7.7kB per game: ```--flatten -l expanded_label -p clark_storkey_2014```
  * Flattened data, both features and labels packed using numpy.packbits
     * 3.2kB per game: ```--flatten -l expanded_label_packed -p clark_storkey_2014_packed```
  * Flattened data, features packed using numpy.packbits, label just one class number
     * 2.9kB per game: ```--flatten -l simple_label -p clark_storkey_2014_packed```


Requirements
------------
 * [gomill library](https://github.com/mattheww/gomill)
 * [hdf5 library](http://www.h5py.org/) if you wish to make Datasets 
