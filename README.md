# KeypointRCNN-GraFormer

This repo contains the PyTorch implementation for Hand object pose and shape estimation using Keypoint RCNN and GraFormer.

## Dependencies

manopth==0.0.1
matplotlib==3.3.4
numpy==1.13.3
opencv_python==4.5.3.56
Pillow==9.2.0
pymeshlab==2021.10
scikit_image==0.17.2
scipy==0.19.1
skimage==0.0
torch==1.10.1
torchvision==0.11.2
tqdm==4.62.3

## Step 1: Create dataset files
datasets/make_data.py creates the dataset files. <br>
Adapt the variables (root and mano_root) to the HO3D dataset path (downloadable through [HO3D](https://www.tugraz.at/index.php?id=40231)) and mano models path (downloadable through [MANO](https://mano.is.tue.mpg.de/))

```
mkdir datasets/ho3d/
python3 datasets/make_data.py
```
# Train default model

```
python3 train_rcnn.py \
  --input_file ../HOPE/datasets/ho-v3-mesh/ \
  --output_file ./checkpoints/ho-rcnn-hand-mesh-3d/model- \
  --train \
  --val \
  --batch_size 1 \
  --gpu_number 1 \
  --object \  
```
This command has the path to the prepared dataset files 
