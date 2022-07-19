import torch 
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import argparse
import matplotlib
matplotlib.use('gtk3agg')
import matplotlib.pyplot as plt

from utils.dataset import Dataset
from utils.vis_utils import *
from tqdm import tqdm
from models.keypoint_rcnn import keypointrcnn_resnet50_fpn
from utils.utils import calculate_keypoints, save_calculate_error, save_dicts, prepare_data_for_evaluation

def collate_fn(batch):
    return tuple(zip(batch))

def visualize2d(img, predictions, labels=None, filename=None, num_keypoints=21, palm=None):
    
    fig = plt.figure(figsize=(25, 15))
    H = 2
    W = 4

    fig_config = (fig, H, W)
    idx = list(predictions['labels']).index(1) #[0]
    hand_faces, obj_faces = load_faces()

    # Plot GT bounding boxes
    if labels is not None:
        plot_bb_ax(img, labels, fig_config, 1, 'GT BB')

        # Plot GT 2D keypoints
        plot_pose2d(img, labels, 0, palm, fig_config, 2, 'GT 2D pose')

        # Plot GT 3D Keypoints
        plot_pose3d(labels, 0, palm, num_keypoints, fig_config, 3, 'GT 3D pose')

        # Plot GT 3D mesh
        plot_mesh3d(labels, 0, palm, num_keypoints, hand_faces, obj_faces, fig_config, 4, 'GT 3D mesh')

    # Plot predicted bounding boxes
    plot_bb_ax(img, predictions, fig_config, 5, 'Predicted BB')
    
    # Plot predicted 2D keypoints
    plot_pose2d(img, predictions, idx, palm, fig_config, 6, 'Predicted 2D pose')

    # Plot predicted 3D keypoints
    plot_pose3d(predictions, idx, palm, num_keypoints, fig_config, 7, 'Predicted 3D pose')

    # Plot predicted 3D Mesh
    plot_mesh3d(predictions, idx, palm, num_keypoints, hand_faces, obj_faces, fig_config, 8, 'Predicted 3D mesh')
    
    # Save Mesh
    save_mesh(predictions, idx, num_keypoints, filename, hand_faces, obj_faces)
    
    fig.tight_layout()
    plt.show()
    # plt.savefig(filename)
    plt.close(fig)

# Input parameters
parser = argparse.ArgumentParser()

# Loading dataset    
parser.add_argument("--split", default='train', help="Which subset to evaluate on")
parser.add_argument("--batch_size", type=int, default=1, help="Mini-batch size")
parser.add_argument("--root", default='./datasets/ho/', help="Dataset root folder")
parser.add_argument("--checkpoint_folder", default='ho', help="the folder of the pretrained model")
parser.add_argument("--checkpoint_id", type=int, required=True, help="the id of the pretrained model")
parser.add_argument("--gpu_number", type=int, nargs='+', default = [1], help="Identifies the GPU number to use.")
parser.add_argument("--hdf5_path", default='', help="Path to HDF5 files to load to the memory for faster training, only suitable for sufficient memory")
parser.add_argument("--seq", default='MPM13', help="Sequence Name")
parser.add_argument("--object", action='store_true', help="generate pose or shape for object?")
parser.add_argument("--visualize", action='store_true', help="Visualize results?")
parser.add_argument("--ycb_path", default='./datasets/ycb_models/', help="Input YCB models, directory")
parser.add_argument("--num_features", type=int, default=2048, help="Number of features passed to coarse-to-fine network")

args = parser.parse_args()

# Transformer function
transform_function = transforms.Compose([transforms.ToTensor()])

init_num_keypoints, num_keypoints = calculate_keypoints(args.object)

# Dataloader
testset = Dataset(root=args.root, load_set=args.split, transform=transform_function, num_keypoints=num_keypoints)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=16, collate_fn=collate_fn)
print(len(testloader.dataset))
print('Data loaded!')

use_cuda = False
if torch.cuda.is_available():
    use_cuda = True

# Define device
device = torch.device(f'cuda:{args.gpu_number[0]}' if torch.cuda.is_available() else 'cpu')

# Define model
model = keypointrcnn_resnet50_fpn(pretrained=False, init_num_kps=init_num_keypoints, num_keypoints=num_keypoints, num_classes=2, device=device,
                                rpn_post_nms_top_n_train=1, rpn_post_nms_top_n_test=1, 
                                # box_detections_per_img=1,
                                box_score_thresh=0.0,
                                # rpn_batch_size_per_image=1, 
                                num_features=args.num_features)

if torch.cuda.is_available():
    model = model.cuda(device=args.gpu_number[0])
    model = nn.DataParallel(model, device_ids=args.gpu_number)

### Load model
pretrained_model = f'./checkpoints/{args.checkpoint_folder}/model-{args.checkpoint_id}.pkl'
model.load_state_dict(torch.load(pretrained_model, map_location='cuda:1'))
model = model.eval()
print(model)
print('model loaded!')

keys = ['boxes', 'labels', 'keypoints', 'keypoints3d', 'mesh3d', 'palm']
c = 0
# supporting_dicts = (pickle.load(open('./rcnn_outputs/rcnn_outputs_778_test_3d.pkl', 'rb')),
#                     pickle.load(open('./rcnn_outputs_mesh/rcnn_outputs_778_test_3d.pkl', 'rb')))
supporting_dicts = None
output_dicts = ({}, {})
errors = []

for i, ts_data in tqdm(enumerate(testloader)):
        
    data_dict = ts_data
    path = data_dict[0][0]['path']
    if args.seq not in path:
        continue

    ### Run inference
    inputs = [t[0]['inputs'].to(device) for t in data_dict]
    outputs = model(inputs)
    img = inputs[0].cpu().detach().numpy()
    
    predictions, img, palm, labels = prepare_data_for_evaluation(data_dict, outputs, img, keys, device, args.split)
    # print(path)
    # print(predictions['scores'][0])
    ### Visualization
    if args.visualize:# and predictions['scores'][0] < 0.05:
        # print(predictions['scores'][0])
        name = path.split('/')[-1]
        if 1 in predictions['labels'] or (1 in predictions['labels'] and 2 in predictions['labels'] and args.object):
            visualize2d(img, predictions, labels, filename=f'./visual_results/{args.seq}/{name}', num_keypoints=num_keypoints, palm=palm)
        else:
            print(predictions['labels'], name)
            cv2.imwrite(f'./visual_results/{args.seq}/{name}', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    ### Evaluation
    c = save_calculate_error(path, predictions, labels, args.split, errors, output_dicts, c, supporting_dicts=supporting_dicts)

if args.split != 'test':
    avg_error = np.average(np.array(errors))
    print('hand pose average error on validation set:', avg_error)

save_dicts(output_dicts, args.split)

