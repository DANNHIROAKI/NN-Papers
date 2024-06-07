#!/usr/bin/env bash
python3 $MG_LEARNING_PATH/src/main/baseline_train.py \
  --id baseline \
  --features 3 \
  --network LightNetworkEnc \
  --conv_nonlinearity elu \
  \
  --train_datasets $1 \
  --train_data_path $MG_TRAIN_DATA \
  \
  --atom_types_path $MG_METADATA_PATH/protein_atom_types.txt \
  --include_near_native \
  --normalize_adj \
  --normalize_x \
  --res_seq_sep 6 \
  --threads 1 \
  \
  --optim adam \
  --lr 0.001 \
  --loss mse \
  --epochs 40 \
  --train_size 512 \
  --batch_size 1 \
  --l2_reg 0.0 \
  \
  --checkpoints $MG_CHECKPOINTS_PATH \
  --bad_targets $MG_METADATA_PATH/bad_targets.txt
