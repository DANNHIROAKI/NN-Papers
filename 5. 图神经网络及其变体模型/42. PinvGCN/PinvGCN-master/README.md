# PinvGCN
Source code for our paper ["Pseudoinverse Graph Convolutional Networks: Fast Filters Tailored for Large Eigengaps of Dense Graphs and Hypergraphs,"](doi.org/10.1007/s10618-021-00752-w) published in Springer Data Mining and Knowledge Discovery (2021).

To use this code, install the required Python packages `torch` and `torch_geometric` and run `python setup.py build` and `python setup.py install`.

All results in the paper were generated with the scripts in the `experiments` directory. For usage, see the following instructions:
- For point clouds: `python run-pointcloud.py -h`
- For hypergraphs: `python run-hypergraph.py -h`
- For sparse graphs: `python run-graph.py -h`
