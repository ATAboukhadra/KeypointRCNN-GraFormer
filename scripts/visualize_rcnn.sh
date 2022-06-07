export CUDA_DEVICE_ORDER=PCI_BUS_ID

python3 vis_rcnn.py \
 --root ../HOPE/datasets/ho-v3-mesh/ \
 --ycb_path ./datasets/ycb_models/ \
 --checkpoint_folder ho-rcnn-hand \
 --checkpoint_id 3 \
 --gpu \
 --gpu_number 1 \
 --split val \
 --seq rgb \
 --graformer \
#  --visualize