#!/usr/bin/env bash
checkpoint_id=baseline
python3 $MG_LEARNING_PATH/src/main/baseline_evaluate.py \
  --checkpoints $MG_CHECKPOINTS_PATH/$checkpoint_id \
  --dataset $1 \
  --data $MG_TEST_DATA \
  --atom_types_path $MG_METADATA_PATH/protein_atom_types.txt \
  --prediction_output $MG_PREDICTIONS_PATH \
  --evaluation_output $MG_RESULTS_PATH \
  --encoder 3 \
  --message_passing 8 \
  --scorer 3 \
  --threads 1 \
  --res_seq_sep 6