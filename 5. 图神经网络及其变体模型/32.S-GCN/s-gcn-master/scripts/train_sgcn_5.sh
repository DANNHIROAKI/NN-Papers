#!/bin/bash

CASPS=$1
python3 $MG_LEARNING_PATH/src/main/sgcn_train.py \
  --id sgcn_5 \
  --features 3 \
  --network S5AndDropout \
  --conv_nonlinearity elu \
  \
  --train_datasets $CASPS \
  --train_data_path $MG_TRAIN_DATA \
  \
  --atom_types_path $MG_METADATA_PATH/protein_atom_types.txt \
  --include_near_native \
  --normalize_x \
  --normalize_adj \
  --sh_order 5 \
  --threads 4 \
  \
  --optim adam \
  --lr 0.001 \
  --dropout 0.02 \
  --l2_reg 0.003 \
  --loss mse \
  --epochs 40 \
  --train_size 2048 \
  --batch_size 64 \
  --shuffle \
  \
  --checkpoints $MG_CHECKPOINTS_PATH \
  --bad_targets $MG_METADATA_PATH/bad_targets.txt
