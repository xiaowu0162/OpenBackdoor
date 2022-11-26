#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=1


TASK_NAME=sst2
MAX_TRAIN_DATA=1000

# --model_name_or_path "textattack/bert-base-uncased-imdb"
# use LR ~1e-5 if start from imdb; use LR ~5e-5 if from pre-trained BERT

python train_sst2_subset_for_kd_teacher.py \
  --model_name_or_path "bert-base-uncased" \
  --task_name $TASK_NAME \
  --do_train \
  --max_train_samples ${MAX_TRAIN_DATA} \
  --do_eval \
  --evaluation_strategy "epoch" \
  --save_strategy "epoch" \
  --save_total_limit 0 \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 5e-5 \
  --num_train_epochs 20 \
  --load_best_model_at_end \
  --metric_for_best_model "accuracy" \
  --output_dir "${TASK_NAME}_${MAX_TRAIN_DATA}data_checkpoints"


rm -r ${TASK_NAME}_${MAX_TRAIN_DATA}data_checkpoints/checkpoint*
