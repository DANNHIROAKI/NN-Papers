This repository contains the source code of the paper "A Hierarchical Entity Graph Convolutional Network for Relation Extraction across Documents" published in RANLP 2021.

### Datasets ###

The dataset used for experiments in the paper can be downloaded from the following link:

https://drive.google.com/drive/folders/1nBima4wLXSRHtMfcMdcQgu3AFlMe9AWP?usp=sharing

### Requirements ###

1) python3.6
2) pytorch 1.7
3) CUDA 8.0

### How to run ###

python3.6 models.py source_dir embedding_file target_dir model_id train

embedding_file = path for glove.6B.300d.txt

python3.6 models.py source_dir embedding_file target_dir model_id test threshold

Above threshold value will be one of outputs of the training process

Use model_id as 1 for CNN, 2 for BiLSTM, 3 for BiLSTM_CNN, 4 for LinkPath, and 5 for our HEGCN model.



