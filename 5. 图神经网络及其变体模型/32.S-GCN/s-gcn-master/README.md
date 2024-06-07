# Spherical graph convolutional networks (S-GCN)

The project contains Spherical Graph Convolutional Network (S-GCN) that processes 3D models of proteins represented as molecular graphs.

#### Content:
1. [Dependencies](#dependencies)
    - [Voronota](#voronota)
    - [Spherical harmonics featurizer](#spherical-harmonics-featurizer)
    - [NOLB](#nolb)
    - [Requirements](#requirements) 
2. [Initialization](#initialization)
3. [Usage](#usage)
    - [Basic example](#basic-example)
    - [Command line arguments](#command-line-arguments)
    - [Versions](#versions)
    - [Output](#output)
4. [Data](#data)
    - [Downloading](#downloading)
    - [Building graphs](#building-graphs)
    - [Adding global scores](#adding-global-scores)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Baseline](#baseline)
8. [Frequent problems](#frequent-problems)

## Dependencies

#### Voronota
S-GCN uses [Voronota](https://github.com/kliment-olechnovic/voronota) by Kliment Olechnovic (kliment@ibt.lt) in order to construct the tessellation and build the molecular graph. 
Prior to running S-GCN, Voronota must be installed. You can use already compiled executables:
* [Linux](./bin/voronota-linux)
* [MacOS](./bin/voronota-macos)

If you want to build the **training data**, you will need to compute ground-truth CAD-scores. This can be done using the special component of the Voronota: `voronota-cadscore`.
In this case you will need to [install](https://github.com/kliment-olechnovic/voronota) the full Voronota library, and then pass the path of the `voronota-cadscore` to the `scripts/init_*.sh` file.

#### Spherical harmonics featurizer
In order to build local coordinate systems and spherical harmonics, S-GCN uses the Spherical harmonics featurizer written in C++.
We provide compiled executables: 
* [Linux](./bin/sh-featurizer-linux)
* [MacOS](./bin/sh-featurizer-macos) 
    
#### NOLB
If you want to add near-native conformations to the data, you will need to download the [NOLB](https://team.inria.fr/nano-d/software/nolb-normal-modes/) tool and provide the path of the executable file `NOLB` as a corresponding argument to the script that builds the data.
Check also the file `scripts/init_*.sh`, there is a commented line that sets a `$MG_NOLB_PATH` variable.

#### Requirements

The main part of the code is written in Python using PyTorch framework. Please, check [requirements](./requirements.txt) for installing necessary Python packages. 

## Initialization
Before running any code, you need to initialize all nesessary variables and create necessary directories.
You can do it with the following commands for Linux:
```
export MG_LEARNING_PATH=/path/to/this/project
. $MG_LEARNING_PATH/scripts/init_linux.sh
```
and for MacOS:
```
export MG_LEARNING_PATH=/path/to/this/project
. $MG_LEARNING_PATH/scripts/init_macos.sh
```
In some cases you might need to check the file `scripts/init_*.sh` itself and fix some paths if they are changed (e.g., if you installed Voronota by yourself). 
Or, for example, uncomment the line with `$MG_NOLB_PATH` if you installed NOLB and you are planning to use it.

## Usage

#### Basic example
If you want to use the ready S-GCN method and predict scores for a model, just specify the path to the model PDB file after the `-i` flag, pass the path of Voronota executable file after the `-v` flag and pass the path of the sh-featurizer executable file after the `-M` flag:
```
python sgcn.py -i /path/to/model.pdb -v /path/to/voronota -M /path/to/sh-featurizer
```
If you want to predict scores for multiple models, specify the path to the directory with model PDB files in argument `-i`, pass the path of Voronota executable file in argument `-v` and pass the path of the sh-featurizer executable file after the `-M` flag: 
```
python sgcn.py -i /path/to/models/ -v /path/to/voronota -M /path/to/sh-featurizer
```
S-GCN will create a folder `sgcn_output/` with the results in the current directory.

Example of the model PDB file that can be passed to the flag `-i` is [here](./casp/test.pdb).

#### Command line arguments

```
Name                 Type    Description                                                     Default
-------------------- ------- --------------------------------------------------------------- ------------------
-i, --input          string  path to the input PDB file or to the directory with PDB files      
-v, --voronota       string  path to Voronota executable file
-M, --maps-generator string  path to sh-featurizer executable file
-o, --output         string  path to the output directory                                    ./
-r  --sgcn-root      string  relative path to the S-GCN project                              ./
-m, --model-version  string  name of S-GCN version                                           sgcn_5_casp_8_11 
-g, --gpu            int     number of GPU device (if present)                               None
-k, --keep-graph     flag    flag to keep graph data for each model in the output directory  False 
-V, --verbose        flag    flag to print all logs to stdout (including warnings)           False
-h, --help           flag    flag to print usage help to stdout and exit
```

#### Versions

Versions that can be passed to the `sgcn.py` with the flag `-m` are trained networks. 
We provide 5 versions that can be used:
* [S-GCN(5)](./versions/sgcn_5_casp_8_11): `sgcn_5_casp_8_11`
* [S-GCN<sub>s</sub>(5)](./versions/sgcn_scoring_5_casp_8_11): `sgcn_scoring_5_casp_8_11`
* [S-GCN<sub>r</sub>(5)](./versions/sgcn_radial_5_casp_8_11): `sgcn_radial_5_casp_8_11`
* [S-GCN(10)](./versions/sgcn_10_casp_8_11): `sgcn_10_casp_8_11`
* [S-GCN<sub>s</sub>(10)](./versions/sgcn_scoring_10_casp_8_11): `sgcn_scoring_10_casp_8_11`

#### Output

S-GCN creates a folder `sgcn_output/` in the output directory. 
By default, in the `sgcn_output/` folder S-GCN creates a file `log` where it writes all logs and a file with extension `.scores` where it writes predicted local scores of the input model.
In case of multiple models, S-GCN creates multiple separate `.score`-files for each input model.
If the flag `-k` is enabled, S-GCN creates for each model its own folder where it writes a `.score`-file and keeps a folder `graph/` with the following content: 
- `graph/model.pdb` - preprocessed input PDB file (H-atoms removed, added chain ID if missed) 
- `graph/x.txt` - file with atom features (one line per atom, the order of atoms coincides with the order in the input PDB file)
- `graph/x_res.txt` - file with residue features (one line per residue, the order of residues coincides with the order the in input PDB file)
- `graph/adj_b.txt` - file with atom-level covalent edges of the graph in format `first_atom second_atom contact_area`<sup>*</sup>
- `graph/adj_c.txt` - file with atom-level contact edges of the graph in format `first_atom second_atom contact_area`<sup>*</sup>
- `graph/adj_res.txt` - file with residue-level contact edges of the graph in format `first_residue second_residue contact_area`<sup>*</sup>
- `graph/covalent_types.txt` - file with atom covalent bonds types in format `first_atom second_atom covalent_type`<sup>*</sup>
- `graph/sequence_separation.txt` - file with atom sequence separation values in format `first_atom second_atom sequence_separation_value`<sup>*</sup>
- `graph/aggr.txt` - file that contains number of atoms in residues (the order of residues coincides with the order in the input PDB file)
- `graph/sh.npy` - binary file that contains spherical harmonics matrices for all residues up to the needed order 

In addition, S-GCN writes to the `stdout` the status of the execution and the predicted global CAD-score of the input model. 
If the flag `-V` is enabled, S-GCN also writes to `stdout` all warnings (they are always written to the `log` file). 

<sup>*</sup> Due to the symmetry of the adjacency matrix, we keep only one edge for each pair of atoms.

## Data

For training and evaluation we use models generated by [CASP](https://predictioncenter.org) challenge participants.
In general, the data is supposed to be kept in the following order:
```
- CASP1
    - T0001 # example of the target name
        - <model 1>
            - model.pdb
            - x.txt
            - x_res.txt
            - adj_b.txt
            - adj_c.txt
            - adj_res.txt
            - covalent_types.txt
            - sequence_separation.txt
            - aggr.txt
            - sh.npy
            - y.txt
        - <model 2>
            ...
        - <model N>
            ...
        - target.pdb
        - log
    - T0002 # example of the target name
        ...
    - T0003 # example of the target name
        ...
- CASP2
    - T0001 # example of the target name
        - <model 1>
            - model.pdb
            - x.txt
            - x_res.txt
            - adj_b.txt
            - adj_c.txt
            - adj_res.txt
            - covalent_types.txt
            - sequence_separation.txt
            - aggr.txt
            - sh.npy
            - y.txt
        - <model 2>
            ...
        - <model N>
            ...
        - target.pdb
        - log
    - T0002 # example of the target name
        ...
    - T0003 # example of the target name
        ...
...
```
We recommend to name folders with CASP data as `CASP<number>/`, and folders with targets as `T<number>/`. 
Depending on the arguments that are used for building the data, some files can be absent, e.g., `target.pdb`, `y.txt` or `y_global.txt`.

For training, files `target.pdb` and `y.txt` are required.

For testing, you do not need to have files `target.pdb` and `y.txt`, but you need to have global scores for each models which should be written in the file `y_global.txt` (for each model one file with one score in the corresponding folder). See section [Adding global scores](#adding-global-scores).

#### Downloading

In order to download the data from one CASP challenge, you can use the [script](./casp/download.sh). 
For example,
```
bash casp/download.sh CASP12
```
This script will create a directory `data/CASP12/` with directories `models/` and `targets/`. 
Directory `data/CASP12/models/` will contain folders with pdb-files of all models associated with corresponding targets.
Directory `data/CASP12/targets/` will contain pdb-files of all targets.     

For quick testing the method, one can use a small portion of data we already downloaded (instead of downloading the full data from CASP website). 
For that, please do the following:
1. Go to `$MG_LEARNING_PATH`
2. Run the following:
```
tar -xzvf casp/CASP12small.tar.gz -C data
mv data/CASP12small data/CASP12
```

#### Building graphs

Once data is downloaded, you can start building graphs. A general entrypoint is a Python script `src/main/preprocess_casp.py`.
It has multiple arguments specifying, for example, whether to compute ground-truth CAD-scores and spherical harmonics.
Example of the script that sets all parameters that are necessary for building the training data is here: `scripts/build_train_data.sh`.
```
bash scripts/build_train_data.sh CASP12
```
Example of the script that sets all parameters that are necessary for building the testing data is here: `scripts/build_test_data.sh`.
```
bash scripts/build_test_data.sh CASP12
```
Always check that you [initialized all variables](#initialization).
Arguments of `src/main/preprocess_casp.py`:

```
Name                   Type    Description                                                Default
--------------------   ------- ---------------------------------------------------------- ---------
--models               string  path to CASP models pdb-files
--targets              string  path to CASP targets pdb-files
--output               string  path to the output directory
--bond_types           string  path to ./metadata/bond_types.csv
--atom_types           string  path to ./metadata/protein_atom_types.txt
--elements_radii       string  path to ./metadata/elements_radii.txt
--voronota_radii       string  path to ./metadata/voronota_radii.txt
--voronota             string  path to Voronota executable
--include_near_native  flag    whether to generate near-native conformations for targets  False
--nolb_rmsd            float   NOLB parameter                                             0.9
--nolb_samples_num     int     NOLB parameter (number of conformations per one target)    50 
--nolb                 string  path to the NOLB executable                                None
--cadscore             string  path to voronota-cadscore executable                       None
                               (set it if you want to compute ground-truth)
--cadscore_window      int     voronota-cadscore parameter                                2
--cadscore_neighbors   int     voronota-cadscore parameter                                1
--sh_featurizer        string  path to sh-featurizer executable                           None
                               (set it if you want to compute spherical harmonics)
--sh_order             int     order of expansion                                         5
--threads              int     number of threads                                          1
```

We emphasize that for further training, the data should contain files `target.gdb` and `y.txt`. 
Please put attention that the arguments connected with `cadscore` are passed correctly.
Don't forget to generate spherical harmonics by passing arguments `-sh_featurizer` and `-sh_order`.

#### Adding global scores

Global CAD-scores can be obtained from the [CASP official website](https://predictioncenter.org).
We also provide [global CAD-scores](./casp/metrics_from_casp.tar.gz) for CASP[10-13] in the archive `casp/metrics_from_casp.tar.gz`.
In order to add these scores in a proper format, extract `casp/metrics_from_casp.tar.gz` and run the [script](./src/main/add_global_scores.py):
```
python $MG_LEARNING_PATH/src/main/add_global_scores.py /path/to/folder/with/casp/for/testing /path/to/unarchived/metrics_from_casp
```

Example for CASP12:
1. Make sure that `$MG_TEST_DATA/CASP12` is not empty (it contains a computed graph for at leat one model)
2. Go to `$MG_LEARNING_PATH` and run the following:
```
tar -xzvf casp/metrics_from_casp.tar.gz -C casp
python src/main/add_global_scores.py $MG_TEST_DATA/CASP12 casp/metrics_from_casp
```
3. Check that now models contain files `y_global.txt`

## Training

Once graphs are built (for example, for CASP8, CASP9, CASP10 and CASP11), you can train a network. 
A general entrypoint is a Python script `src/main/sgcn_train.py`.
Example of its usage for training S-GCN(5) is [here](./scripts/train_sgcn_5.sh):
```
bash scripts/train_sgcn_5.sh CASP8,CASP9,CASP10,CASP11
```

The result of the training is a directory `checkpoints/` which contains a folder with all training checkpoints.
For example, any of these checkpoints can be moved to the directory `versions/` and used for prediction.

Check that your data contains spherical harmonics `sh.npy` and ground-truth `y.txt`!

Arguments of the entrypoint `src/main/sgcn_train.py`:

```
Name                         Type    Description                                                    Default
---------------------------- ------- -------------------------------------------------------------- ---------
--id                         string  id (name) of the training model
--features                   int     number of features (it always should be 3)
--network                    string  name of the network architecture (see src/sgcn/networks.py)
--conv_nonlinearity          string  name of non-linearity inside spherical conv. layers

--train_datasets             string  comma-separated CASPs used for training (e.g., CASP8,CASP9)
--train_data_path            string  path to the training data (e.g., ./train_graphs)
--train_gemme_features_path

--atom_types_path            string  path to ./metadata/protein_atom_types.txt     
--include_near_native        flag    whether include near-native models while training               False
--normalize_x                flag    whether normalize features                                      False
--normalize_adj              flag    whether normalize adj matrix S                                  False
--include_contacts           flag    flag for training S-GCN_r (with radial components)              False
--sh_order                   int     order of expansion                                              5
--shuffle                    flag    whether to shuffle data                                         False
--threads                    int     number of parallel threads                                      1
--gpu                        int     number of the GPU-device (if present)                           None

--optim                      string  name of optimizer ('adam', 'sgd')
--lr                         float   learning rate
--l2_reg                     float   L2-refularization parameter                                     0.0
--dropout                    float   dropout probability value                                       0.0
--loss                       string  loss function (it always should be 'mse')
--epochs                     int     number of epochs                                                15  
--train_size                 int     number of models that are processed by one thread               None
                                     in one epoch (if None then the whole data is processed)
--batch_size                 int     batch size                                                      64
--memory_size                int     number of models that are loaded in memory by one thread        512

--checkpoints                string  path to the directory where checkpoints will be saved
--log_path                   string  path to the log                                                 None
--bad_targets                string  path to the file with bad targets (that will be ignored)        None
```

## Evaluation

In order to evaluate the trained network (for example, on the CASP12), you can use a Python script `src/main/sgcn_evaluate.py`.
Example of its usage for evaluation S-GCN(5) is [here](./scripts/evaluate_sgcn_5.sh):
```
bash scripts/evaluate_sgcn_5.sh CASP12
```

The result of the script is two directories: `predictions/` and `results/`. 
The directory `predictions/` contains predicted local scores of models for all checkpoints, and `results/` contains a table with metrics for all checkpoints. 

We should ephasize that if files `y.txt` or `y_global.txt` are not present, then instead of them predictions will be used. 
We recommend to check whether all models in the test datatset contain `y_global.txt` and `y.txt`, or use the list of targets for which ground-truth score can be computed (check tables with this targets in our Supplementary Mateerials).

```
Name                         Type    Description                                                    Default
---------------------------- ------- -------------------------------------------------------------- --------- 
--checkpoints                string  path to the directory with checkpoints of the network
--dataset                    string  name of the dataset (e.g., CASP13)
--data                       string  path to the testing data (e.g., ./test_graphs)
--atom_types_path            string  path to ./metadata/protein_atom_types.txt
--prediction_output          string  path to directory where to save [predictions] 
                                     (e.g., ./predictions)
--evaluation_output          string  path to directory where to save results (e.g., ./results)
--gpu                        int     number of the GPU-device (if present)                           None
--threads                    int     number of parallel threads                                      1
--epochs                     string  comma-separated epoch which you want to evaluate                None
                                     (if None, all epochs are evaluated)
--predict_only               flag    whether only predict without evaluation                         False
```


The state-of-the-art results can be obtained from the [CASP official website](https://predictioncenter.org). We provide these results in the [archive](./casp/state_of_the_art.tar.gz) `casp/state_of_the_art.tar.gz`.

## Baseline
In order to train and evaluate the baseline network, you can use the same Python scripts as for S-GCN networks
by replacing `sgcn` to `baseline` in filenames.

To use these scripts you can chose other numbers of encoder, message-passing, and scorer layers with the
following arguments:
```
Name                 Type    Description                          Default
-------------------- ------- ------------------------------------ ------------------
--encoder            int     number of encoder layers             3
--message_passing    int     number of message-passing layers     8      
--scorer             int     number of scorer layers              3
```

An example of training and evaluating of baseline network you can find in `scripts/train_baseline.sh` and `scripts/evaluate_baseline.sh`.

## Frequent problems

1. General advise: do not forget to [initialize all variables](#initialization)!
2. In case of missing some files in the built data, please check that you ran the correct script:
* `scripts/build_train_data.sh` – for bulding the training data (usually used for CASP[8-11])
* `scripts/build_test_data.sh` – for testing data (usually used for CASP[12-13]): it does not work with original target structure and does not compute CAD-scores. As a result, you will not see the following files in the resulting directory:
    - `target.pdb`
    - `y.txt`
    - `y_global.txt`

3. For Mac users it might be usefult to clean directories with `find . -name ".DS_Store" -delete`.
