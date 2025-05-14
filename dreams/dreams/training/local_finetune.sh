#!/bin/bash

#python download_data.py

python3 dreams/training/train.py \
 --project_name denovo_dataset_v1 \
 --job_key "job_rt_finetune" \
 --run_name "run_rt_finetune" \
 --train_objective RTINSECONDS \
 --train_regime fine-tuning \
 --dataset_pth "../data/train.hdf5" \
 --dformat A \
 --model DreaMS \
 --ff_peak_depth 1 \
 --ff_fourier_depth 2 \
 --ff_fourier_d 32 \
 --ff_out_depth 1 \
 --n_layers 1 \
 --n_heads 2 \
 --d_peak 12 \
 --d_fourier 32 \
 --lr 3e-5 \
 --batch_size 2 \
 --prec_intens 1.1 \
 --num_devices 1 \
 --max_epochs 2 \
 --log_every_n_steps 20 \
 --head_depth 1 \
 --seed 3407 \
 --train_precision 32 \
 --pre_trained_pth "..." \
 --val_check_interval 0.1 \
 --max_peaks_n 60 \
 --save_top_k -1 \
 --dropout 0.1 \
 --att_dropout 0.1 \
 --residual_dropout 0.1 \
 --ff_dropout 0.1 \
 --weight_decay 0 \
 --attn_mech dot-product \
 --no_transformer_bias \
 --n_warmup_steps 5000 \
 --fourier_strategy lin_float_int \
 --mz_shift_aug_p 0.2 \
 --mz_shift_aug_max 50 \
 --pre_norm \
 --graphormer_mz_diffs \
 --wandb_entity_name j-vangoey \
 --num_workers_data 8 \
 --no_wandb

