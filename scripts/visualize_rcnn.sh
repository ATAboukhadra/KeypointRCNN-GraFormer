export CUDA_DEVICE_ORDER=PCI_BUS_ID

python3 vis_rcnn.py \
 --root ./datasets/ho3d/ \
 --ycb_path ./datasets/ycb_models/ \
 --checkpoint_folder hand-mesh \
 --checkpoint_id 11 \
 --split test \
 --seq rgb \
#  --visualize
#  --object \
