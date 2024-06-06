#!/bin/bash
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_standalone_train_ascend.sh dataset dataroot device_id noBN output_path"
echo "For example: bash run_standalone_train_ascend.sh lsun /opt_data/xidian_wks/lsun 3 False output_dir"
echo "It is better to use the absolute path."
echo "=============================================================================================================="

EXEC_PATH=$(pwd)
echo "$EXEC_PATH"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}
DATA_PATH=$(get_real_path $2)
OUT_PATH=$(get_real_path $5)
export DEVICE_ID=$3

cd ../
rm -rf train
mkdir train
cd ./train
mkdir src
cd ../
cp ./*.py ./train
cp ./src/*.py ./train/src
cd ./train

env > env0.log

echo "train begin."
python train.py --dataset $1 --dataroot $DATA_PATH --device_id $3 --noBN $4 --experiment $OUT_PATH > ./train.log 2>&1 &

