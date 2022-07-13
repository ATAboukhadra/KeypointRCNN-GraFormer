#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID

python3 train_rcnn.py \
  --input_file ../HOPE/datasets/ho-v3-mesh/ \
  --output_file ./checkpoints/ho-rcnn-hand-mesh-3d/model- \
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
  --gpu \
  # --object \  
#   --generate_mesh \