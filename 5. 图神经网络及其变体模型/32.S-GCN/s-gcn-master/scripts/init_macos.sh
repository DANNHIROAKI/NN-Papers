#!/bin/bash

export MG_VORONOTA_PATH=$MG_LEARNING_PATH/bin/voronota-macos
export MG_SH_FEATURIZER_PATH=$MG_LEARNING_PATH/bin/sh-featurizer-macos
#export MG_VORONOTA_CADSCORE_PATH=<put your path to the compiled voronota-cadscore file>
#export MG_NOLB_PATH=<put your path to the compiled NOLB file>

export MG_METADATA_PATH=$MG_LEARNING_PATH/metadata

export MG_RAW_DATA=$MG_LEARNING_PATH/data
export MG_TRAIN_DATA=$MG_LEARNING_PATH/train_graphs
export MG_TEST_DATA=$MG_LEARNING_PATH/test_graphs
export MG_CHECKPOINTS_PATH=$MG_LEARNING_PATH/checkpoints
export MG_PREDICTIONS_PATH=$MG_LEARNING_PATH/predictions
export MG_RESULTS_PATH=$MG_LEARNING_PATH/results
export MG_CHEKPOINTS_EPOCH_PREFIX=checkpoint_epoch_

mkdir -p $MG_TRAIN_DATA
mkdir -p $MG_TEST_DATA
mkdir -p $MG_CHECKPOINTS_PATH
mkdir -p $MG_PREDICTIONS_PATH
mkdir -p $MG_RESULTS_PATH
