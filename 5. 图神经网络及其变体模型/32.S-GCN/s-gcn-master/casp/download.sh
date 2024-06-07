#!/bin/bash

function unpack {
  CASP_dataset=$1
  native_structures=$2
  decoy_structures=$3

  mkdir -p data/$CASP_dataset/models
  mkdir -p data/$CASP_dataset/targets
  for file in predictioncenter.org/download_area/$CASP_dataset/$decoy_structures
  do
    tar -xzf $file -C data/$CASP_dataset/models 2>> errorlog.txt
  done

  if [ $CASP_dataset == 'CASP7' ]; then
    for structure in data/$CASP_dataset/models/*
    do
      mv $structure data/$CASP_dataset/models/T$(basename $structure) 2>> errorlog.txt
    done
  fi

  mkdir data/$CASP_dataset/temp
  tar -xzf predictioncenter.org/download_area/$CASP_dataset/$native_structures -C data/$CASP_dataset/temp 2>> errorlog.txt

  if [ $CASP_dataset == 'CASP7' ]; then
    mv data/$CASP_dataset/temp/TARGETS/* data/$CASP_dataset/temp/
    rmdir data/$CASP_dataset/temp/TARGETS
  fi

  rm data/$CASP_dataset/temp/*_* 2>> errorlog.txt
  rm data/$CASP_dataset/temp/T[^0]* 2>> errorlog.txt

  for native in data/$CASP_dataset/temp/*
  do
    native_name=$(basename ${native%.pdb})
    mv $native data/$CASP_dataset/targets/$native_name.pdb 2>> errorlog.txt
  done
  rm -r data/$CASP_dataset/temp
}

CASP_dataset=$1
echo $CASP_dataset...

if [ $CASP_dataset == 'CASP13' ]; then
  native_structures='targets/casp13.targets.T.4public.tar.gz'
  wget -r -A tar.gz --no-parent http://predictioncenter.org/download_area/$CASP_dataset/server_predictions/
  wget -r -A $(basename $native_structures) --no-parent http://predictioncenter.org/download_area/$CASP_dataset/targets/
  unpack $CASP_dataset $native_structures "server_predictions/*.stage2.3D.srv.tar.gz"
fi

if [ $CASP_dataset == 'CASP12' ]; then
  native_structures='targets/casp12.targets_T0.releaseDec022016.tgz'
  wget -r -A tar.gz --no-parent http://predictioncenter.org/download_area/$CASP_dataset/server_predictions/
  wget -r -A $(basename $native_structures) --no-parent http://predictioncenter.org/download_area/$CASP_dataset/targets/
  unpack $CASP_dataset $native_structures "server_predictions/?????.stage2.3D.srv.tar.gz"
fi

if [ $CASP_dataset == 'CASP11' ]; then
  native_structures='targets/casp11.targets_unsplitted.release11242014.tgz'
  wget -r -A tar.gz --no-parent http://predictioncenter.org/download_area/$CASP_dataset/server_predictions/
  wget -r -A $(basename $native_structures) --no-parent http://predictioncenter.org/download_area/$CASP_dataset/targets/
  unpack $CASP_dataset $native_structures "server_predictions/?????.stage2.3D.srv.tar.gz"
fi

if [ $CASP_dataset == 'CASP10' ]; then
  native_structures='targets/casp10.targets_unsplitted.noT0695T0739.tgz'
  wget -r -A tar.gz --no-parent http://predictioncenter.org/download_area/$CASP_dataset/server_predictions/
  wget -r -A $(basename $native_structures) --no-parent http://predictioncenter.org/download_area/$CASP_dataset/targets/
  unpack $CASP_dataset $native_structures "server_predictions/?????.stage2.3D.srv.tar.gz"
fi

if [ $CASP_dataset == 'CASP9' ]; then
  native_structures='targets/casp9.targ_unsplit.tgz'
  wget -r -A tar.gz --no-parent http://predictioncenter.org/download_area/$CASP_dataset/server_predictions/
  wget -r -A $(basename $native_structures) --no-parent http://predictioncenter.org/download_area/$CASP_dataset/targets/
  unpack $CASP_dataset $native_structures "server_predictions/*"
fi

if [ $CASP_dataset == 'CASP8' ]; then
  native_structures='targets/casp8.targ_unsplit.tar.gz'
  wget -r -A tar.gz --no-parent http://predictioncenter.org/download_area/$CASP_dataset/server_predictions/
  wget -r -A $(basename $native_structures) --no-parent http://predictioncenter.org/download_area/$CASP_dataset/targets/
  unpack $CASP_dataset $native_structures "server_predictions/*"
fi

if [ $CASP_dataset == 'CASP7' ]; then
  native_structures='targets/targets.all.tgz'
  wget -r -A tar.gz --no-parent http://predictioncenter.org/download_area/$CASP_dataset/server_predictions/
  wget -r -A $(basename $native_structures) --no-parent http://predictioncenter.org/download_area/$CASP_dataset/targets/
  unpack $CASP_dataset $native_structures "server_predictions/*"
fi

if [ $CASP_dataset == 'CASP6' ]; then
  native_structures='targets_all.tgz'
  wget -r -A tar.gz --no-parent http://predictioncenter.org/download_area/$CASP_dataset/MODELS_SUBMITTED/
  wget -r -A $native_structures --no-parent http://predictioncenter.org/download_area/$CASP_dataset/
  unpack $CASP_dataset $native_structures "MODELS_SUBMITTED/*"
  rm data/$CASP_dataset/*/T????[ADRS]*
fi

if [ $CASP_dataset == 'CASP5' ]; then
  native_structures='casp5_targ.tar.gz'
  wget -r -A tar.gz --no-parent http://predictioncenter.org/download_area/$CASP_dataset/MODELS_SUBMITTED/
  wget -r -A $native_structures --no-parent http://predictioncenter.org/download_area/$CASP_dataset/
  unpack $CASP_dataset $native_structures "MODELS_SUBMITTED/*"
  rm data/$CASP_dataset/*/T????[ADRS]*
fi
