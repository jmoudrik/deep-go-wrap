# deep-go-wrap
Toolkit designed to ease development of your Deep Neural Network models
for the game of Go.

  * **[CNN Dataset Construction](#hdf)** - making HDF5 datasets for your Convolutional Neural Networks.
  * **[GTP wrapper](#gtp)** - wrapper turning your CNN to standalone [GTP](http://www.lysator.liu.se/~gunnar/gtp/) player.

<a name="hdf"></a>HDF Go Datasets
------------------
  * different planes support, easy to extend
     * [Clark and Storkey 2014](http://arxiv.org/abs/1412.3409)
     * [Tian and Zhu 2015](http://arxiv.org/abs/1511.06410)
     * planes from DeepCL
     * others (e.g. Detlef Schmicker's 54%)
  * parallel processing of games
  * the HDF dataset created is compatible with pylearn2 for instance, but NOT with DeepCL. To create dataset for DeepCL, see [hdf2deepcl_v2.py](hdf2deepcl_v2.py) tool.

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
Running times were basically the same (~ 1.8 sec per game on commodity laptop). HDF5 compresses
the dataset transparently using gzip -9. The feature cube is composed of 7 planes
used in [Clark an Storkey's 2014 paper](http://arxiv.org/pdf/1412.3409).
The ```--flatten``` option just reshapes data from ```(7,19,19)``` to ```(7*19*19,)```.

  * Fully expanded, flattened data in floats32 (e.g. directly ready for pylearn2)
     * 14.2kB per game: ```--dtype float32 --flatten -l expanded_label -p clark_storkey_2014```
  * Fully expanded, flattened data in uint8
     * 7.7kB per game: ```--flatten -l expanded_label -p clark_storkey_2014```
  * Flattened data, both features and labels packed using numpy.packbits
     * 3.2kB per game: ```--flatten -l expanded_label_packed -p clark_storkey_2014_packed```
  * Flattened data, features packed using numpy.packbits, label just one class number
     * 2.9kB per game: ```--flatten -l simple_label -p clark_storkey_2014_packed```

<a name="gtp"></a>GTP wrapper
-----------------------------
 * interface for plugging in various Deep Network architectures
    * caffe nets, DeepCL nets (*working*), easy to extend
 * I/O handling, data planes extraction
 * full GTP support using gomill library
 * pass/resign implementation, using GnuGo as an oracle. Note that this slows thigs down a bit, GnuGo being slower than CNN.
 * move correctness checking
 * **Do you have other great ideas? Contribute, or make an issue!**

#### caffe network setup
1. Adding up new net is easy. Here I use Detlef Schmicker's model as an example -- see [deepgo/bot_caffe.py](/deepgo/bot_caffe.py).
2. You can get (Detlef's 54% CNN)[http://physik.de/CNNlast.tar.gz]

3. Edit path to prototxt and trained model file in the [deepgowrap.py](deepgowrap.py) (you may have to uncomment main_deepcl in the code)
    * now the deepgowrap acts as an GTP program, so you can run it:

    ```echo -e "boardsize 19\nclear_board B\nquit" | ./deepgowrap.py```

#### DeepCL setup (slightly obsolete)
1. Train your deepcl model
    1. get data https://github.com/hughperkins/kgsgo-dataset-preprocessor
    2. get DeepCL, probably [my fork](https://github.com/jmoudrik/DeepCL) of [Hugh Perkins's repo](https://github.com/hughperkins/DeepCL)
    which is sure to work, the changes aren't big and will probably be merged to the 
    Hugh's repo, so this is for the time being.
    3. train your model
    4. or, instead of the last step, just download my test small weights (precision 19.6%, trained in 10 minutes, extremely weak, from [here](http://j2m.cz/~jm/weights.dat) (binary file, might not work on different archs), trained with:

        ```./deepclrun numtrain=200000 dataset=kgsgoall 'netdef=32c3{z}-relu-4*(8c3{z}-relu)-128n-tanh-361n' numepochs=20 learningrate=0.0001```
2. edit path to DeepCL and model's weights file in the [deepgowrap.py](deepgowrap.py) (you may have to uncomment main_deepcl in the code)
    * now the deepgowrap acts as an GTP program, so you can run it:

    ```echo -e "boardsize 19\nclear_board B\nquit" | ./deepgowrap.py```


Requirements
------------
 * [gomill library](https://github.com/mattheww/gomill)
 * [hdf5 library](http://www.h5py.org/) if you wish to make Datasets 
