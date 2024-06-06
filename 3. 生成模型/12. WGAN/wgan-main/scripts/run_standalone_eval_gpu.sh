#!/bin/bash
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_standalone_eval_gpu.sh device_id config ckpt_file output_dir nimages"
echo "For example: bash run_standalone_eval_gpu.sh DEVICE_ID CONFIG_PATH CKPT_FILE_PATH OUTPUT_DIR NIMAGES"
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

CONFIG_PATH=$(get_real_path $2)
CKPT_FILE_PATH=$(get_real_path $3)
OUTPUT_DIR=$(get_real_path $4)

mkdir src
cd ../
rm -rf eval
mkdir eval
cd ./eval
mkdir src
cd ../
cp ./*.py ./eval
cp ./src/*.py ./eval/src
cd ./eval

env > env0.log

echo "eval begin."
python eval.py --device_id $1 --config $CONFIG_PATH --ckpt_file $CKPT_FILE_PATH --output_dir $OUTPUT_DIR --nimages $5 --device_target GPU> ./eval.log 2>&1 &

