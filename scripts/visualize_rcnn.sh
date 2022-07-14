export CUDA_DEVICE_ORDER=PCI_BUS_ID

python3 vis_rcnn.py \
 --root ../HOPE/datasets/ho-v3-mesh/ \
 --ycb_path ./datasets/ycb_models/ \
 --checkpoint_folder hand-mesh \
 --checkpoint_id 1 \
 --split train \
 --seq rgb \
 --num_features 2048 \
 --visualize
#  --object \
