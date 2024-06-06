#!/bin/bash
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_distributed_train_ascend.sh dataset dataroot device_num rank_tabel_file noBN output_path"
echo "For example: bash run_distributed_train_ascend.sh lsun /opt_data/xidian_wks/lsun 8 hccl_8p_01234567_127.0.0.1.json False dis_output_dir"
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

export DATASET=$DATA_PATH
export OUTPUT_PATH=$OUTPUT_PATH
export RANK_TABLE_FILE=$RANK_TABLE_FILE
export DEVICE_NUM=$4
export RANK_SIZE=$4
export HCCL_CONNECT_TIMEOUT=600
export SERVER_ID=0
rank_start=$((DEVICE_NUM * SERVER_ID))

# remove old train_parallel files
rm -rf train_dis
mkdir train_dis
echo "device count=$DEVICE_NUM"

i=0
while [ $i -lt ${DEVICE_NUM} ]; do
    export DEVICE_ID=${i}
    export RANK_ID=$((rank_start + i))

    # mkdirs
    mkdir ./train_dis/$i
    mkdir ./train_dis/$i/src

    # move files
    cp ../*.py ./train_parallel/$i
    cp ../src/*.py ./train_parallel/$i/src

    # goto the training dirs of each training
    cd ./train_parallel/$i || exit

    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    # input logs to env.log
    env > env.log
    python -u train.py --dataset $1 --dataroot $DATA_PATH --noBN $5 --experiment $OUT_PATH  --device_target Ascend --distribute=True --ckpt_dir=./ckpt --dataset=$DATASET --output_path=$OUTPUT_PATH > log 2>&1 &
    cd ../..
    i=$((i + 1))
done
