#!/bin/bash
python sample.py \
--num_epochs 32 --batch_size 16 \
--num_D_steps 1 --G_lr 2.5e-5 --D_lr 2.5e-5 --D_B2 0.999 --G_B2 0.999 \
--G_ch 32 --D_ch 32 \
--G_attn 64 --D_attn 64 \
--G_nl inplace_Lrelu --D_nl inplace_relu \
--G_shared \
--hier --dim_z 256 --shared_dim 128 \
--SN_eps 1e-8 --BN_eps 1e-5 --adam_eps 1e-8 \
--G_ortho 1e-4 \
--G_init ortho --D_init ortho \
--G_eval_mode \
--ema --use_ema --ema_start 80000 \
--save_every 5000 --num_save_copies 0 --seed 1234 \
--sample_num 10000 --trunc_z 0.8
