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

def draw_confidence(image, keypoints, scores):
    keypoints = np.round(keypoints).astype(np.int)

    high_confidence = np.where(scores >= 2)[0]
    low_confidence = np.where(scores < 2)[0]
    # print(high_confidence)
    
    for idx in high_confidence:
        cv2.circle(image, center=(keypoints[idx][0], keypoints[idx][1]), radius=3, color=[43, 140, 237], thickness=-1)
    for idx in low_confidence:
        cv2.circle(image, center=(keypoints[idx][0], keypoints[idx][1]), radius=3, color=[0, 0, 0], thickness=-1)
    
    return image

def visualize2d(img, predictions, labels=None, filename=None, mesh=False, object=False):
    
    # print(n)
    # img = img.astype(np.uint8)
    fig = plt.figure(figsize=(25, 15))

    width = 2
    height = 2
    i = 1
    hand_idx = list(predictions['labels']).index(1) #[0]
    if object:
        obj_idx = list(predictions['labels']).index(2) #[0]

    # boxes = predictions['boxes']
    # if bb_hand is not None:
    #     bb_image = draw_bb(original_img, bb_hand, [229, 255, 204])
    #     bb_image = draw_bb(bb_image, bb_object, [204, 229, 255])
    # else:
    #     bb_image = original_img
    bb_image = np.copy(img)
    if labels is not None:
        ax = fig.add_subplot(height, width, 1)
        ax.title.set_text('GT BB')

        bb_hand = labels['boxes'][0]
        # bb_object = labels['boxes'][1]
        # print(bb_hand)
        bb_image = draw_bb(bb_image, bb_hand, [229, 255, 204])
        # bb_image = draw_bb(bb_image, bb_object, [204, 229, 255])
        ax.imshow(bb_image)

    ax = fig.add_subplot(height, width, 2)
    ax.title.set_text('Predicted BB')

    bb_image = np.copy(img)
    # draw_confidence(bb_image, predictions['keypoints'][hand_idx], predictions['keypoints_scores'][hand_idx])
    for bb in predictions['boxes']:
        # bb_hand = predictions['boxes'][0]
        # bb_object = predictions['boxes'][1]
    #     # print(bb_hand)
        bb_image = draw_bb(bb_image, bb, [229, 255, 204])
        # bb_image = draw_bb(bb_image, bb_object, [204, 229, 255])
    ax.imshow(bb_image)

    # Plot 2D annotations
    gt_image = np.copy(img)
    if labels is not None:
        hand_keypoints = labels['keypoints'][0]
        if object:
            obj_keypoints = labels['keypoints'][1]
        
        ax = fig.add_subplot(height, width, 3)
        if mesh:
            if object:
                keypoints = np.append(hand_keypoints[:778], obj_keypoints, axis=0)
            else:
                keypoints = hand_keypoints[:778]

            show2DMesh(fig, ax, gt_image, keypoints, filename=filename)
        else:
            gt_image = showHandJoints(gt_image, hand_keypoints)
        # gt_image = showObjJoints(gt_image, obj_keypoints[:8])
            ax.imshow(gt_image)
        ax.title.set_text('GT keypoints')

    pred_image = np.copy(img)
    # print(type(predictions['labels']))
    # object_idx = list(predictions['labels']).index(2)#[0]
    # print(hand_idx, object_idx)
    
    hand_keypoints = predictions['keypoints'][hand_idx]
    if object:
        obj_keypoints = predictions['keypoints'][obj_idx]
    # object_box = predictions['boxes'][object_idx]
    
    # print(hand_keypoints.shape)
    # projected_keypoints2d = project_3D_points(cam_mat, keypoints3d, is_OpenGL_coords=False)
    ax = fig.add_subplot(height, width, 4)
    if mesh:
        if object:
            keypoints = np.append(hand_keypoints[:778], obj_keypoints, axis=0)
        else:
            keypoints = hand_keypoints[:778]        # show2DMesh(fig, ax, pred_image, keypoints, filename=filename)
        show2DMesh(fig, ax, pred_image, keypoints)

    else:
        pred_image = showHandJoints(pred_image, hand_keypoints, filename=filename)
        # pred_image = showHandJoints(pred_image, labels['keypoints'][0], filename=filename, mode='gt')

        ax.imshow(pred_image)

    # img = showHandJoints(img, predicted_keypoints2d[:21], dataset_name=dataset_name)
    # pred_image = showObjJoints(pred_image, obj_keypoints[:8])
    # pred_image = draw_bb(pred_image, object_box, [229, 255, 204])
    ax.title.set_text('Predicted keypoints')


    # # Plot 2D joint predictions
    # # projected_predicted_keypoints2d = project_3D_points(cam_mat, predicted_keypoints3d, is_OpenGL_coords=False)
    # img = showHandJoints(original_img, predicted_keypoints2d[:21], dataset_name=dataset_name)
    
    # img = showObjJoints(img, predicted_keypoints2d[21:], filename=file_path)
    # ax = fig.add_subplot(height, width, i + width)
    # ax.imshow(img)
    # ax.title.set_text('predicted 2D joints')

    # i += 1
    # # Plot 3D keypoint Annotations
    # coordChangeMat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
    # # if isOpenGLCoords:

    # if keypoints3d is not None:
    #     ax = fig.add_subplot(height, width, i, projection="3d")
    #     show3DHandJoints(ax, keypoints3d[:21], mode='gt', isOpenGLCoords=False)
    #     show3DHandJoints(ax, predicted_keypoints3d[:21], isOpenGLCoords=False)    
        
    #     show3DObjCorners(ax, keypoints3d[21:], mode='gt', isOpenGLCoords=False)
    #     show3DObjCorners(ax, predicted_keypoints3d[21:], isOpenGLCoords=False)
        
    #     cam_equal_aspect_3d(ax, keypoints3d, flip_x=False)
    #     ax.title.set_text('3D ground truth and predictions\n (black=ground truth)')

    # ax = fig.add_subplot(height, width, i + width, projection="3d")
    # show3DHandJoints(ax, predicted_keypoints3d[:21], isOpenGLCoords=False)
    # show3DObjCorners(ax, predicted_keypoints3d[21:], isOpenGLCoords=False)
    # cam_equal_aspect_3d(ax, predicted_keypoints3d, flip_x=False)
    # ax.title.set_text('predicted 3D joints')
    
    # i += 1

    fig.tight_layout()
    # plt.subplots_adjust(wspace=0.2, hspace=0.3)
    # plt.show()
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
parser.add_argument("--gpu", action='store_true', help="Switch for gpu computation.")
parser.add_argument("--crop", action='store_true', help="Crop images around the center")
parser.add_argument("--hdf5_path", default='', help="Path to HDF5 files to load to the memory for faster training, only suitable for sufficient memory")
parser.add_argument("--seq", default='MPM13', help="Sequence Name")
parser.add_argument("--generate_mesh", action='store_true', help="Generate 3D mesh")
parser.add_argument("--object", action='store_true', help="generate pose or shape for object?")
parser.add_argument("--visualize", action='store_true', help="Visualize results?")
parser.add_argument("--graformer", action='store_true', help="Add graformer to Mask RCNN")
parser.add_argument("--ycb_path", default='./datasets/ycb_models/', help="Input YCB models, directory")

args = parser.parse_args()

# Transformer function
transform_function = transforms.Compose([transforms.ToTensor()])

testset = Dataset(root=args.root, load_set=args.split, transform=transform_function, return_mesh=args.generate_mesh, object=args.object)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=16, collate_fn=collate_fn)
print(len(testloader.dataset))
print('Data loaded!')

keypoints = 21
if args.generate_mesh:
    keypoints = 1000

use_cuda = False
if args.gpu:
    use_cuda = True

# Define device
device = torch.device(f'cuda:{args.gpu_number[0]}' if torch.cuda.is_available() else 'cpu')

model = keypointrcnn_resnet50_fpn(pretrained=False, num_keypoints=keypoints, num_classes=2, device=device, add_graformer=args.graformer)

if args.gpu and torch.cuda.is_available():
    if args.graformer:
        model.roi_heads.keypoint_graformer.mask = model.roi_heads.keypoint_graformer.mask.cuda(args.gpu_number[0])
    model = model.cuda(device=args.gpu_number[0])
    model = nn.DataParallel(model, device_ids=args.gpu_number)

pretrained_model = f'./checkpoints/{args.checkpoint_folder}/model-{args.checkpoint_id}.pkl'
model.load_state_dict(torch.load(pretrained_model, map_location='cuda:1'))
model = model.eval()
print(model)
print('model loaded!')

minLoss = 100000
criterion = nn.MSELoss()
keys = ['boxes', 'labels', 'keypoints']
c_hand, c_obj = 0, 0
output_dict = {}
errors = []

# for i, ts_data in tqdm(enumerate(testloader)):

for i, ts_data in tqdm(enumerate(testloader)):
        
    data_dict = ts_data

    path = data_dict[0][0]['path']
    # print(path)
    if args.seq not in path:
        continue
    # wrap them in Variable
    targets = [{k: v.to(device) for k, v in t[0].items() if k in keys} for t in data_dict]
    inputs = [t[0]['inputs'].to(device) for t in data_dict]
    # original_input = data_dict['original_image'].cpu().detach().numpy()[0]
    
    outputs = model(inputs)

    img = inputs[0].cpu().detach().numpy()
    labels = {k: v.cpu().detach().numpy() for k, v in targets[0].items()}
    predictions = {k: v.cpu().detach().numpy() for k, v in outputs[0].items()}

    path = data_dict[0][0]['path']
    name = path.split('/')[-1]

    if args.split == 'test':
        labels = None

    # if 'AP11' in path:
    img = img.transpose(1, 2, 0) * 255
    img = np.ascontiguousarray(img, np.uint8) 

    ### Visualization

    if args.visualize:
        if 1 in predictions['labels'] or (1 in predictions['labels'] and 2 in predictions['labels'] and args.object):
        #     # visualize2d(img, predictions, labels, filename=f'./visual_results/{args.seq}_GT/{name}', mesh=args.generate_mesh)
            visualize2d(img, predictions, labels, filename=f'./visual_results/{args.seq}/{name}', mesh=args.generate_mesh, object=args.object)
        else:
            print(predictions['labels'], name)
            cv2.imwrite(f'./visual_results/{args.seq}/{name}', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    ### Evaluation
    predicted_labels = list(predictions['labels'])
    
    if 1 in predicted_labels:
        hand_idx = predicted_labels.index(1) 
        hand_keypoints = predictions['keypoints'][hand_idx]
    
        if args.split != 'test':
            hand_keypoints_gt = labels['keypoints'][0]
            error = mpjpe(torch.Tensor(hand_keypoints[:, :2]), torch.Tensor(hand_keypoints_gt[:, :2]))
            errors.append(error)

    else:
        c_hand += 1
        hand_keypoints = np.zeros((21, 3))
        # print(c_hand)
      
#     if 2 in predicted_labels:
#         obj_idx = predicted_labels.index(2) 
#         obj_keypoints = predictions['keypoints'][obj_idx][:8]
#         confidence = predictions['keypoints_scores'][obj_idx][:8]
#         obj_keypoints[:, 2] = confidence
#     else:
#         c_obj += 1
#         obj_keypoints = np.zeros((8, 3))
# #         print(c_hand, c_obj)
    if args.object:
        keypoints = np.append(hand_keypoints, obj_keypoints, axis=0)
    else:
        keypoints = hand_keypoints
    output_dict[path] = keypoints
#     # print(keypoints)
    
# #     # if 1 in predictions['labels'] and 2 in predictions['labels']:
# #     #     visualize2d(img, predictions, labels)

if args.split != 'test':
    avg_error = np.average(np.array(errors))
    print('average error:', avg_error)

output_dict = dict(sorted(output_dict.items()))

with open(f'rcnn_outputs_hand_{args.split}.pkl', 'wb') as f:
    pickle.dump(output_dict, f)
