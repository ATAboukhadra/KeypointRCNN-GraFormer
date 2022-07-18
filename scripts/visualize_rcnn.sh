export CUDA_DEVICE_ORDER=PCI_BUS_ID

python3 vis_rcnn.py \
 --root ../HOPE/datasets/ho-v2-mesh/ \
 --ycb_path ./datasets/ycb_models/ \
 --checkpoint_folder photometric-2048 \
 --checkpoint_id 42 \
 --gpu \
 --gpu_number 1 \
 --split test \
 --seq SB13 \
 --graformer \
 --object \
 --generate_mesh \
 --visualize
