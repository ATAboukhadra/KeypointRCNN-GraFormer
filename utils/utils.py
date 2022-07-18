import torch
import numpy as np
import pickle

def calculate_bounding_box(point2d, increase=False):
    pad_size = 15
    x_min = int(min(point2d[:,0]))
    y_min = int(min(point2d[:,1]))
    x_max = int(max(point2d[:,0]))
    y_max = int(max(point2d[:,1]))
    
    if increase:
        return np.array([x_min - pad_size, y_min - pad_size, x_max + pad_size, y_max + pad_size])
    else:
        return np.array([x_min, y_min, x_max, y_max])


def create_rcnn_data(bb, point2d, point3d, num_keypoints=21):
    ''' Prepares data for an RCNN by creating tensors for Bounding boxes, labels and keypoints with their visibility'''
            
    # Boxes and Labels
    boxes = torch.Tensor(bb[np.newaxis, ...]).float()
    labels = torch.from_numpy(np.array([1])) # 1 for hand-object box or hand-only box
    
    # Appending visibility TODO: change this to actual visibility
    visibility = np.ones(num_keypoints).reshape(-1, 1)
    keypoints = np.append(point2d[:num_keypoints], visibility, axis=1)

    # Append keypoints
    final_keypoints = torch.Tensor(keypoints[:num_keypoints][np.newaxis, ...]).float()
    final_keypoints3d = torch.Tensor(point3d[:num_keypoints][np.newaxis, ...]).float()

    return boxes, labels, final_keypoints, final_keypoints3d

def freeze_component(model):
    for param in model.parameters():
        param.requires_grad = False
    
def calculate_keypoints(obj):

    num_keypoints = 1778 if obj else 778
    init_num_kps = 21 if num_keypoints == 778 else 29

    return init_num_kps, num_keypoints

def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))

def save_calculate_error(path, predictions, labels, split, errors, output_dicts, c, supporting_dicts=None, rgb_errors=None, img=None):
    """Stores the results of the model in a dict and calculates error in case of available gt"""

    
    predicted_labels = list(predictions['labels'])

    if 1 in predicted_labels:
        idx = predicted_labels.index(1) 
        keypoints = predictions['keypoints3d'][idx][:21]
        mesh = predictions['mesh3d'][idx]
        if split != 'test':
            mesh_gt = labels['mesh3d'][0][:778]
            error = mpjpe(torch.Tensor(mesh[:778, :3]), torch.Tensor(mesh_gt))
            errors.append(error)
            rgb_error = calculate_rgb_error(img, labels['mesh3d'][0], mesh[:, 3:])
            rgb_errors.append(rgb_error)

    else:
        c += 1
        if supporting_dicts is not None:
            keypoints = supporting_dicts[0][path]
            mesh = supporting_dicts[1][path]
        else:
            keypoints = np.zeros((21, 3))
            mesh = np.zeros((778, 3))
        print(c)
      
    output_dicts[0][path] = keypoints
    output_dicts[1][path] = mesh

    return c

def save_dicts(output_dicts, split):
    
    output_dict = dict(sorted(output_dicts[0].items()))
    output_dict_mesh = dict(sorted(output_dicts[1].items()))
    print('Total number of predictions:', len(output_dict.keys()))

    with open(f'./rcnn_outputs/rcnn_outputs_21_{split}_3d_v2.pkl', 'wb') as f:
        pickle.dump(output_dict, f)

    with open(f'./rcnn_outputs_mesh/rcnn_outputs_778_{split}_3d_v2.pkl', 'wb') as f:
        pickle.dump(output_dict_mesh, f)

def prepare_data_for_evaluation(data_dict, outputs, img, keys, device, split):
    """Postprocessing function"""

    targets = [{k: v.to(device) for k, v in t[0].items() if k in keys} for t in data_dict]

    labels = {k: v.cpu().detach().numpy() for k, v in targets[0].items()}
    predictions = {k: v.cpu().detach().numpy() for k, v in outputs[0].items()}


    palm = labels['palm'][0]
    if split == 'test':
        labels = None

    img = img.transpose(1, 2, 0) * 255
    img = np.ascontiguousarray(img, np.uint8) 

    return predictions, img, palm, labels

def project_3D_points(pts3D):

    cam_mat = np.array(
        [[617.343,0,      312.42],
        [0,       617.343,241.42],
        [0,       0,       1]])

    proj_pts = pts3D.dot(cam_mat.T)
    proj_pts = np.stack([proj_pts[:,0] / proj_pts[:,2], proj_pts[:,1] / proj_pts[:,2]], axis=1)
    # proj_pts = proj_pts.to(torch.long)
    return proj_pts


def generate_gt_texture(image, mesh3d):
    mesh2d = project_3D_points(mesh3d)

    image = image / 255

    H, W, _ = image.shape

    idx_x = mesh2d[:, 0].clip(min=0, max=W-1).astype(np.int)
    idx_y = mesh2d[:, 1].clip(min=0, max=H-1).astype(np.int)

    texture = image[idx_y, idx_x]
    
    return texture

def calculate_rgb_error(image, mesh3d, p_texture):
    texture = generate_gt_texture(image, mesh3d)
    error = mpjpe(torch.Tensor(texture), torch.Tensor(p_texture))
    return error
