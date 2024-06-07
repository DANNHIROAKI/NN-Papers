#!/bin/bash

python3 $MG_LEARNING_PATH/sgcn.py \
  --input $1 \
  --voronota $MG_VORONOTA_PATH \
  --maps-generator $MG_MAPS_GENERATOR_PATH \
  --output $2 \
  --model-version sgcn_5_casp_8_11 \
  --keep-graph \
  --sgcn-root $MG_LEARNING_PATH \
  --verbose
