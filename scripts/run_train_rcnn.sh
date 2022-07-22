#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID

python3 train_rcnn.py \
  --input_file ./datasets/ho3d/ \
  --output_file ./checkpoints/photometric-2048/model- \
  --train \
  --val \
  --batch_size 1 \
  --gpu_number 1 \
  --learning_rate 0.0001 \
  --lr_step 100 \
  --lr_step_gamma 0.9 \
  --log_batch 100 \
  --val_epoch 1 \
  --snapshot_epoch 1 \
  --num_iterations 50 \
  --num_features 2048 \
  --object \
