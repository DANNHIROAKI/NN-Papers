#!/bin/bash

CASP=$1
python $MG_LEARNING_PATH/src/main/preprocess_casp.py \
    --models $MG_RAW_DATA/${CASP}/models \
    --output $MG_TEST_DATA/${CASP} \
    --bond_types $MG_METADATA_PATH/bond_types.csv \
    --atom_types $MG_METADATA_PATH/protein_atom_types.txt \
    --elements_radii $MG_METADATA_PATH/elements_radii.txt \
    --voronota_radii $MG_METADATA_PATH/voronota_radii.txt \
    --voronota $MG_VORONOTA_PATH \
    --threads 1 \
    \
    --sh_featurizer $MG_SH_FEATURIZER_PATH \
    --sh_order 5
