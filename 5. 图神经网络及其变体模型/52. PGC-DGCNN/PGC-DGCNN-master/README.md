Installation
------------

This implementation is based on Hanjun Dai's PyTorch version of structure2vec and Muhan's implementation

Please first clone it to this folder by

    git clone https://github.com/Hanjun-Dai/pytorch_structure2vec

Or, alternatively, you can directly unzip the pytorch_structure2vec-master.zip.

Then, under the "pytorch_structure2vec-master/s2vlib/" directory, type

    make -j4

to build the necessary c++ backend.

After that, under the root directory of this repository, type

    ./run_DGCNN.sh
