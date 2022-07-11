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
from manopth.manolayer import ManoLayer

cam_mat = np.array(
    [[617.343,0,      312.42],
    [0,       617.343,241.42],
    [0,       0,       1]
 ])

mano_layer = ManoLayer(mano_root='../HOPE/manopth/mano/models', use_pca=False, ncomps=6, flat_hand_mean=True)
handFaces = mano_layer.th_faces
print("Mano layer faces loaded!")

# Loading object faces
obj_mesh = read_obj('../HOPE/datasets/spheres/sphere_1000.obj')
objFaces = obj_mesh.f


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

def visualize2d(img, predictions, labels=None, filename=None, num_keypoints=21, palm=None):
    
    # print(n)
    # img = img.astype(np.uint8)
    fig = plt.figure(figsize=(25, 15))

    width = 4
    height = 2
    i = 1
    idx = list(predictions['labels']).index(1) #[0]
    
    predicted_keypoints3d = predictions['keypoints3d'][idx] 
    # if palm is not None:
    #     predicted_keypoints3d += palm
        
    projected_keypoints2d = project_3D_points(cam_mat, predicted_keypoints3d + palm, is_OpenGL_coords=False)

    # Plot GT bounding boxes
    bb_image = np.copy(img)
    if labels is not None:
        ax = fig.add_subplot(height, width, 1)
        ax.title.set_text('GT BB')
        bb = labels['boxes'][0]
        bb_image = draw_bb(bb_image, bb, [229, 255, 204])
        ax.imshow(bb_image)

    # Plot predicted bounding boxes
    ax = fig.add_subplot(height, width, 5)
    ax.title.set_text('Predicted BB')

    bb_image = np.copy(img)
    # draw_confidence(bb_image, predictions['keypoints'][hand_idx], predictions['keypoints_scores'][hand_idx])
    for bb in predictions['boxes']:
        bb_image = draw_bb(bb_image, bb, [229, 255, 204])
    ax.imshow(bb_image)

    # Plot GT 2D keypoints
    gt_image = np.copy(img)
    if labels is not None:
        keypoints = labels['keypoints'][0]
        # keypoints = projected_keypoints2d
        
        ax = fig.add_subplot(height, width, 2)
        gt_image = showHandJoints(gt_image, keypoints[:21])
        if keypoints.shape[0] > 21:
            gt_image = showObjJoints(gt_image, keypoints[21:])
        
        ax.imshow(gt_image)
        ax.title.set_text('GT keypoints')

        # Plot GT 3D Keypoints
        keypoints3d = labels['keypoints3d'][0]
        if palm is not None:
            keypoints3d += palm
    
        ax = fig.add_subplot(height, width, 3, projection="3d")
        show3DHandJoints(ax, keypoints3d[:21], mode='gt', isOpenGLCoords=True)
        if num_keypoints > 778:
            show3DObjCorners(ax, keypoints3d[21:], mode='gt', isOpenGLCoords=True)
        
        ax.title.set_text('GT 3D pose')

        # Plot GT 3D mesh
        ax = fig.add_subplot(height, width, 4, projection="3d")
        keypoints3d = labels['mesh3d'][0]
        if palm is not None:
            keypoints3d += palm
        plot3dVisualize(ax, keypoints3d[:778], handFaces, flip_x=False, isOpenGLCoords=False, c="r")
        if num_keypoints > 778:
            plot3dVisualize(ax, keypoints3d[778:], objFaces, flip_x=False, isOpenGLCoords=False, c="b")
        cam_equal_aspect_3d(ax, keypoints3d[:num_keypoints], flip_x=False)
        ax.title.set_text('Original Mesh')
    
        
    # Plot predicted 2D keypoints
    pred_image = np.copy(img)
    # keypoints2d = predictions['keypoints'][idx]
    keypoints2d = projected_keypoints2d

    ax = fig.add_subplot(height, width, 6)     
    pred_image = showHandJoints(pred_image, keypoints2d[:21], filename=filename)
    if keypoints2d.shape[0] > 21:
        pred_image = showObjJoints(pred_image, keypoints2d[21:], filename=filename)
    ax.imshow(pred_image)
    ax.title.set_text('Predicted keypoints')

    # Plot predicted 3D keypoints
    predicted_keypoints3d = predictions['keypoints3d'][idx]
    if palm is not None:
        predicted_keypoints3d += palm

    ax = fig.add_subplot(height, width, 7, projection="3d")
    show3DHandJoints(ax, predicted_keypoints3d[:21], isOpenGLCoords=True)
    if num_keypoints > 778:
        show3DObjCorners(ax, predicted_keypoints3d[21:], isOpenGLCoords=True)
    
    ax.title.set_text('Pred 3D pose')

    # Plot predicted 3D Mesh
    ax = fig.add_subplot(height, width, 8, projection="3d")
    predicted_keypoints3d = predictions['mesh3d'][idx]
    if palm is not None:
        predicted_keypoints3d += palm

    plot3dVisualize(ax, predicted_keypoints3d[:778], handFaces, flip_x=False, isOpenGLCoords=False, c="r")
    final_obj = filename.replace('.jpg', '').replace('.png', '')
    
    if num_keypoints > 778:
        print(final_obj)
        plot3dVisualize(ax, predicted_keypoints3d[778:], objFaces, flip_x=False, isOpenGLCoords=False, c="b")        
        final_faces = np.concatenate((handFaces, objFaces + 778), axis = 0)
        write_obj(predicted_keypoints3d, final_faces, final_obj)
    else:
        write_obj(predicted_keypoints3d, handFaces, final_obj)
    
    cam_equal_aspect_3d(ax, predicted_keypoints3d[:num_keypoints], flip_x=False)
    ax.title.set_text('Predicted Mesh')

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
