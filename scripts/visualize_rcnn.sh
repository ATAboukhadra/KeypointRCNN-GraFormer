export CUDA_DEVICE_ORDER=PCI_BUS_ID

python3 vis_rcnn.py \
 --root ./datasets/ho3d/ \
 --ycb_path ./datasets/ycb_models/ \
 --checkpoint_folder photometric-2048 \
 --checkpoint_id 42 \
 --gpu_number 1 \
 --split val \
 --seq rgb \
#  --visualize
