#!/bin/bash
air_path=${1}
om_path=${2}

atc --input_format=NCHW \
    --framework=1 \
    --model="${air_path}" \
    --output="${om_path}" \
    --soc_version=Ascend310 \
    --precision_mode=allow_fp32_to_fp16 \
    --soc_version=Ascend310 \
    --op_select_implmode=high_precision
