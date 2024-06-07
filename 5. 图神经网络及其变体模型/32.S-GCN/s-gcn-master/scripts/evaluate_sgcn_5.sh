#!/bin/bash

CASP=$1
CHECKPOINT_ID=sgcn_5
python3 $MG_LEARNING_PATH/src/main/sgcn_evaluate.py \
  --checkpoints $MG_CHECKPOINTS_PATH/$CHECKPOINT_ID \
  --dataset $CASP \
  --data $MG_TEST_DATA \
  --atom_types_path $MG_METADATA_PATH/protein_atom_types.txt \
  --prediction_output $MG_PREDICTIONS_PATH \
  --evaluation_output $MG_RESULTS_PATH \
  --threads 1
