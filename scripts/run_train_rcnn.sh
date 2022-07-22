#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID

python3 train_rcnn.py \
  --input_file ./datasets/ho3d/ \
  --output_file ./checkpoints/photometric-2048/model- \
  --train \
  --val \
  --batch_size 1 \
  --gpu_number 1 \
  --object \
