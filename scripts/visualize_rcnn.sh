export CUDA_DEVICE_ORDER=PCI_BUS_ID

python3 vis_rcnn.py \
 --root ../HOPE/datasets/ho-v3-mesh/ \
 --ycb_path ./datasets/ycb_models/ \
 --checkpoint_folder photometric-2048 \
 --checkpoint_id 10 \
 --gpu \
 --gpu_number 1 \
 --split val \
 --seq rgb \
 --graformer \
 --object \
 --generate_mesh \
#  --visualize