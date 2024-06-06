#!/bin/bash
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_distribute_train_gpu.sh dataset dataroot device_num visible_devices noBN output_path"
echo "For example: bash run_distribute_train_gpu.sh lsun /opt_data/xidian_wks/lsun 8 0,1,2,3,4,5,6,7 False dis_output_dir"
echo "It is better to use the absolute path."
echo "=============================================================================================================="

EXEC_PATH=$(pwd)
echo "$EXEC_PATH"

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}
DATA_PATH=$(get_real_path $2)
OUT_PATH=$(get_real_path $6)

ulimit -c unlimited
export DEVICE_NUM=$3
export RANK_SIZE=$3
export CUDA_VISIBLE_DEVICES=$4
cd ../
rm -rf train_dis
mkdir train_dis
cd ./train_dis
mkdir src
cd ../
cp ./*.py ./train_dis
cp ./src/*.py ./train_dis/src
cd ./train_dis

env > env0.log

echo "train_dis begin."
mpirun -n $DEVICE_NUM --output-filename log_output --merge-stderr-to-stdout --allow-run-as-root python train.py --device_target GPU --run_distribute True \
--dataset $1 --dataroot $DATA_PATH --device_target GPU --noBN $5 --experiment $OUT_PATH> train.dis_log 2>&1 &
