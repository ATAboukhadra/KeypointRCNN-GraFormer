export CUDA_DEVICE_ORDER=PCI_BUS_ID

python3 vis_rcnn.py \
 --root ../HOPE/datasets/ho-v3-mesh/ \
 --ycb_path ./datasets/ycb_models/ \
 --checkpoint_folder ho-rcnn-hand-mesh-3d \
 --checkpoint_id 3 \
 --split test \
 --seq rgb \
# --visualize
#  --object \
# --num_features 2048 \

