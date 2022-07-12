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
from GraFormer.common.loss import mpjpe
from models.keypoint_rcnn import keypointrcnn_resnet50_fpn


def collate_fn(batch):
    return tuple(zip(batch))

def visualize2d(img, predictions, labels=None, filename=None, num_keypoints=21, palm=None):
    
    # print(n)
    # img = img.astype(np.uint8)
    fig = plt.figure(figsize=(25, 15))
    H = 2
    W = 4

    fig_config = (fig, H, W)
    idx = list(predictions['labels']).index(1) #[0]
    hand_faces, obj_faces = load_faces()

    # Plot GT bounding boxes
    if labels is not None:
        plot_bb_ax(img, 0, labels, fig_config, 1, 'GT BB')

        # Plot GT 2D keypoints
        plot_pose2d(img, 0, labels, fig_config, 2, 'GT 2D pose')

        # Plot GT 3D Keypoints
        plot_pose3d(labels, 0, palm, num_keypoints, fig_config, 3, 'GT 3D pose')

        # Plot GT 3D mesh
        plot_mesh3d(labels, 0, palm, num_keypoints, hand_faces, obj_faces, fig_config, 4, 'GT 3D mesh')

    # Plot predicted bounding boxes
    plot_bb_ax(img, predictions, idx, fig_config, 5, 'Predicted BB')
    
    # Plot predicted 2D keypoints
    plot_pose2d(img, predictions, idx, fig_config, 6, 'Predicted 2D pose')

    # Plot predicted 3D keypoints
    plot_pose3d(predictions, idx, palm, num_keypoints, fig_config, 7, 'Predicted 3D pose')

    # Plot predicted 3D Mesh
    plot_mesh3d(predictions, idx, palm, num_keypoints, hand_faces, obj_faces, fig_config, 8, 'Predicted 3D mesh')
    
    # Save Mesh
    predicted_keypoints3d = predictions['mesh3d'][idx]
    if num_keypoints > 778:
        final_faces = np.concatenate((hand_faces, obj_faces + 778), axis = 0)
        write_obj(predicted_keypoints3d, final_faces, final_obj)
    else:
        write_obj(predicted_keypoints3d, hand_faces, final_obj)

    fig.tight_layout()
    # plt.subplots_adjust(wspace=0.2, hspace=0.3)
    plt.show()
    # plt.savefig(filename)
    plt.close(fig)

# Input parameters
parser = argparse.ArgumentParser()

# Loading dataset    
parser.add_argument("--split", default='train', help="Which subset to evaluate on")
parser.add_argument("--batch_size", type=int, default=1, help="Mini-batch size")
parser.add_argument("--dimension", type=int, default=3, help="2D or 3D")
parser.add_argument("--root", default='./datasets/ho/', help="Dataset root folder")
parser.add_argument("--checkpoint_folder", default='ho', help="the folder of the pretrained model")
parser.add_argument("--checkpoint_id", type=int, required=True, help="the id of the pretrained model")
parser.add_argument("--gpu_number", type=int, nargs='+', default = [1], help="Identifies the GPU number to use.")
parser.add_argument("--gpu", action='store_true', help="Switch for gpu computation.")
parser.add_argument("--crop", action='store_true', help="Crop images around the center")
parser.add_argument("--hdf5_path", default='', help="Path to HDF5 files to load to the memory for faster training, only suitable for sufficient memory")
parser.add_argument("--seq", default='MPM13', help="Sequence Name")
parser.add_argument("--generate_mesh", action='store_true', help="Generate 3D mesh")
parser.add_argument("--object", action='store_true', help="generate pose or shape for object?")
parser.add_argument("--visualize", action='store_true', help="Visualize results?")
parser.add_argument("--graformer", action='store_true', help="Add graformer to Mask RCNN")
parser.add_argument("--ycb_path", default='./datasets/ycb_models/', help="Input YCB models, directory")
parser.add_argument("--feature_extractor", action='store_true', help="Add feature extractor in Mask RCNN")
args = parser.parse_args()

# Transformer function
transform_function = transforms.Compose([transforms.ToTensor()])

if args.object:
    num_keypoints = 29
else:
    num_keypoints = 21
    
if args.generate_mesh:
    if args.object:
        init_num_keypoints = 29
        num_keypoints = 1778
    else:
        init_num_keypoints = 21
        num_keypoints = 778
else:
    init_num_keypoints = num_keypoints
testset = Dataset(root=args.root, load_set=args.split, transform=transform_function, num_keypoints=num_keypoints)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=16, collate_fn=collate_fn)
print(len(testloader.dataset))
print('Data loaded!')

use_cuda = False
if args.gpu:
    use_cuda = True

# Define device
device = torch.device(f'cuda:{args.gpu_number[0]}' if torch.cuda.is_available() else 'cpu')

model = keypointrcnn_resnet50_fpn(pretrained=False, init_num_kps=init_num_keypoints, num_keypoints=num_keypoints, num_classes=2, device=device,
                                rpn_post_nms_top_n_train=1, rpn_post_nms_top_n_test=1, add_graformer=args.graformer, rpn_batch_size_per_image=1)

if args.gpu and torch.cuda.is_available():
    if args.graformer:
        model.roi_heads.keypoint_graformer.mask = model.roi_heads.keypoint_graformer.mask.cuda(args.gpu_number[0])
        # model.roi_heads.keypoint_graformer2d.mask = model.roi_heads.keypoint_graformer2d.mask.cuda(args.gpu_number[0])

    model = model.cuda(device=args.gpu_number[0])
    model = nn.DataParallel(model, device_ids=args.gpu_number)

pretrained_model = f'./checkpoints/{args.checkpoint_folder}/model-{args.checkpoint_id}.pkl'
model.load_state_dict(torch.load(pretrained_model, map_location='cuda:1'))
model = model.eval()
print(model)
print('model loaded!')

minLoss = 100000
criterion = nn.MSELoss()
keys = ['boxes', 'labels', 'keypoints', 'keypoints3d', 'mesh3d', 'palm']
c = 0


supporting_dict = pickle.load(open('./rcnn_outputs/rcnn_outputs_778_test_3d.pkl', 'rb'))
supporting_dict_mesh = pickle.load(open('./rcnn_outputs_mesh/rcnn_outputs_778_test_3d.pkl', 'rb'))

output_dict = {}
output_dict_mesh = {}

errors = []

for i, ts_data in tqdm(enumerate(testloader)):
        
    data_dict = ts_data

    path = data_dict[0][0]['path']
    if args.seq not in path:
        continue
    # wrap them in Variable
    targets = [{k: v.to(device) for k, v in t[0].items() if k in keys} for t in data_dict]
    # print(targets)
    inputs = [t[0]['inputs'].to(device) for t in data_dict]
    # original_input = data_dict['original_image'].cpu().detach().numpy()[0]
        
    outputs = model(inputs)
    
    img = inputs[0].cpu().detach().numpy()
    labels = {k: v.cpu().detach().numpy() for k, v in targets[0].items()}
    predictions = {k: v.cpu().detach().numpy() for k, v in outputs[0].items()}

    # predictions = {k: v.cpu().detach().numpy() for k, v in targets[0].items()}

    path = data_dict[0][0]['path']
    # print(path)
    name = path.split('/')[-1]

    palm = labels['palm'][0]
    if args.split == 'test':
        labels = None

    # if 'AP11' in path:
    img = img.transpose(1, 2, 0) * 255
    img = np.ascontiguousarray(img, np.uint8) 

    ### Visualization

    if args.visualize:
        if 1 in predictions['labels'] or (1 in predictions['labels'] and 2 in predictions['labels'] and args.object):
        #     # visualize2d(img, predictions, labels, filename=f'./visual_results/{args.seq}_GT/{name}', mesh=args.generate_mesh)
            visualize2d(img, predictions, labels, filename=f'./visual_results/{args.seq}/{name}', num_keypoints=num_keypoints, palm=palm)
        else:
            print(predictions['labels'], name)
            cv2.imwrite(f'./visual_results/{args.seq}/{name}', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    ### Evaluation
    predicted_labels = list(predictions['labels'])
    
    if 1 in predicted_labels:
        idx = predicted_labels.index(1) 
        if args.dimension == 3:
            keypoints = predictions['keypoints3d'][idx][:, :args.dimension]
            mesh = predictions['mesh3d'][idx]
        else:
            keypoints = predictions['keypoints'][idx][:, :args.dimension]
            mesh = np.array([])
    
        if args.split != 'test':
            keypoints_gt = labels['keypoints'][0][:, :args.dimension]
            error = mpjpe(torch.Tensor(keypoints[:21]), torch.Tensor(keypoints_gt[:21]))
            errors.append(error)

    else:
        c += 1
        if supporting_dict is not None:
            keypoints = supporting_dict[path]
            mesh = supporting_dict_mesh[path]
        else:
            keypoints = np.zeros((num_keypoints, args.dimension))
            mesh = np.zeros((num_keypoints, args.dimension))
        print(c)
      
    output_dict[path] = keypoints
    output_dict_mesh[path] = mesh
    # break
# prof.stop()

if args.split != 'test':
    avg_error = np.average(np.array(errors))
    print('hand pose average error on validation set:', avg_error)

output_dict = dict(sorted(output_dict.items()))
print('Total number of predictions:', len(output_dict.keys()))

with open(f'./rcnn_outputs/rcnn_outputs_{num_keypoints}_{args.split}_3d_v2.pkl', 'wb') as f:
    pickle.dump(output_dict, f)

output_dict_mesh = dict(sorted(output_dict_mesh.items()))
with open(f'./rcnn_outputs_mesh/rcnn_outputs_{num_keypoints}_{args.split}_3d_v2.pkl', 'wb') as f:
    pickle.dump(output_dict_mesh, f)
