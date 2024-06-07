# python R2-W1-F1.py --task final/machineryralf --model mlp --scheduler_batch 8 --nreqs 338
# python R2-W1-F1.py --task final/tickralfv2 --model lr --max_error 0.04 --scheduler_batch 1 --nreqs 4740
# python R2-W1-F1.py --task final/tripsralfv2 --model lgbm --max_error 1.5 --scheduler_batch 2 --nreqs 220
# python R2-W1-F1.py --task final/turbofan --model rf --max_error 4.88 --scheduler_batch 9 --nreqs 769
# python R2-W1-F1.py --task final/batteryv2 --model lgbm --max_error 189.0 --scheduler_batch 5 --nreqs 564
# python R2-W1-F1.py --task final/tdfraudralf2d --model xgb --scheduler_batch 3 --nreqs 8603
# python R2-W1-F1.py --task final/studentqnov2subset --model rf --scheduler_batch 13 --nreqs 471
# python R2-W1-F1.py --task final/machineryralfsimmedian0 --model mlp --scheduler_batch 8 --nreqs 338
# python R2-W1-F1.py --task final/tripsralfv3 --model lgbm --max_error 1.4 --scheduler_batch 2 --nreqs 1964

sudo /home/ckchang/anaconda3/envs/apx/bin/python R2-W1-F1.py --task final/machineryralf --model mlp --scheduler_batch 8 --nreqs 338  # ssd17
sudo /home/ckchang/anaconda3/envs/apx/bin/python R2-W1-F1.py --task final/tickralfv2 --model lr --max_error 0.04 --scheduler_batch 1 --nreqs 4740  # ssd8
sudo /home/ckchang/anaconda3/envs/apx/bin/python R2-W1-F1.py --task final/tripsralfv2 --model lgbm --max_error 1.5 --scheduler_batch 2 --nreqs 2016  # ssd16
sudo /home/ckchang/anaconda3/envs/apx/bin/python R2-W1-F1.py --task final/turbofan --model rf --max_error 4.88 --scheduler_batch 9 --nreqs 769  # ssd6
sudo /home/ckchang/anaconda3/envs/apx/bin/python R2-W1-F1.py --task final/batteryv2 --model lgbm --max_error 189.0 --scheduler_batch 5 --nreqs 564  # ssd18
sudo /home/ckchang/anaconda3/envs/apx/bin/python R2-W1-F1.py --task final/tdfraudralf2d --model xgb --scheduler_batch 3 --nreqs 8603  # ssd6
sudo /home/ckchang/anaconda3/envs/apx/bin/python R2-W1-F1.py --task final/studentqnov2subset --model rf --scheduler_batch 13 --nreqs 471  # ssd16
sudo /home/ckchang/anaconda3/envs/apx/bin/python R2-W1-F1.py --task final/machineryralfsimmedian0 --model mlp --scheduler_batch 8 --nreqs 338  # ssd17
sudo /home/ckchang/anaconda3/envs/apx/bin/python R2-W1-F1.py --task final/tripsralfv3 --model lgbm --max_error 1.4 --scheduler_batch 2 --nreqs 1964  # ssd18