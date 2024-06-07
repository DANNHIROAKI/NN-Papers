# !/bin/bash
# interpreter="python"
# interpreter="sudo PYTHONPATH=/home/ckchang/ApproxInfer /home/ckchang/anaconda3/envs/apxinf/bin/python"
interpreter="/home/ckchang/anaconda3/envs/apx/bin/python"

task_name="tick"
agg_qids="1"
sudo $interpreter prep.py --interpreter $interpreter --task_name $task_name --prepare_again --all_nparts 2 100

model="lr"
ncores=1 # only one core by default
nparts=100 # ncfgs=nparts by default
loading_mode=0

shared_opts="--interpreter $interpreter --task_name $task_name --agg_qids $agg_qids --model $model --nparts $nparts --ncores $ncores  --loading_mode $loading_mode"
sudo $interpreter eval_reg.py $shared_opts --run_shared

max_error=1.0
scheduler_init=1
scheduler_batch=1
sudo $interpreter eval_reg.py $shared_opts --scheduler_init $scheduler_init --scheduler_batch $scheduler_batch --max_error $max_error