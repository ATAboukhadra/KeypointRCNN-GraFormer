import torch
import numpy as np
from GraFormer.common.loss import mpjpe

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
    # print(H, W)

    # print(max(mesh2d[:, 0]), max(mesh2d[:, 1]))
    idx_x = mesh2d[:, 0].clip(min=0, max=W-1).astype(np.int)
    idx_y = mesh2d[:, 1].clip(min=0, max=H-1).astype(np.int)

    texture = image[idx_y, idx_x]
    
    return texture

def calculate_rgb_error(image, mesh3d, p_texture):
    texture = generate_gt_texture(image, mesh3d)
    error = mpjpe(torch.Tensor(texture), torch.Tensor(p_texture))
    return error
