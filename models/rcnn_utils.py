import torch
import torch.nn.functional as F
import os
from torchvision.ops import roi_align
from typing import Optional, List, Dict, Tuple
from torch import nn, Tensor

# from pytorch3d.loss import (
#     mesh_edge_loss, 
#     mesh_laplacian_smoothing, 
#     mesh_normal_consistency,
#     chamfer_distance
# )
# from pytorch3d.structures import Meshes
# from pytorch3d.io import load_obj, save_obj

def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN.

    Args:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction='sum',
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss

def maskrcnn_inference(x, labels):
    # type: (Tensor, List[Tensor]) -> List[Tensor]
    """
    From the results of the CNN, post process the masks
    by taking the mask corresponding to the class with max
    probability (which are of fixed size and directly output
    by the CNN) and return the masks in the mask field of the BoxList.

    Args:
        x (Tensor): the mask logits
        labels (list[BoxList]): bounding boxes that are used as
            reference, one for ech image

    Returns:
        results (list[BoxList]): one BoxList for each image, containing
            the extra field mask
    """
    mask_prob = x.sigmoid()

    # select masks corresponding to the predicted classes
    num_masks = x.shape[0]
    boxes_per_image = [label.shape[0] for label in labels]
    labels = torch.cat(labels)
    index = torch.arange(num_masks, device=labels.device)
    mask_prob = mask_prob[index, labels][:, None]
    mask_prob = mask_prob.split(boxes_per_image, dim=0)

    return mask_prob

def project_masks_on_boxes(gt_masks, boxes, matched_idxs, M):
    # type: (Tensor, Tensor, Tensor, int) -> Tensor
    """
    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.
    """
    matched_idxs = matched_idxs.to(boxes)
    rois = torch.cat([matched_idxs[:, None], boxes], dim=1)
    gt_masks = gt_masks[:, None].to(rois)
    return roi_align(gt_masks, rois, (M, M), 1.)[:, 0]

def maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs):
    # type: (Tensor, List[Tensor], List[Tensor], List[Tensor], List[Tensor]) -> Tensor
    """
    Args:
        proposals (list[BoxList])
        mask_logits (Tensor)
        targets (list[BoxList])

    Return:
        mask_loss (Tensor): scalar tensor containing the loss
    """

    discretization_size = mask_logits.shape[-1]
    labels = [gt_label[idxs] for gt_label, idxs in zip(gt_labels, mask_matched_idxs)]
    mask_targets = [
        project_masks_on_boxes(m, p, i, discretization_size)
        for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)
    ]

    labels = torch.cat(labels, dim=0)
    mask_targets = torch.cat(mask_targets, dim=0)

    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if mask_targets.numel() == 0:
        return mask_logits.sum() * 0

    mask_loss = F.binary_cross_entropy_with_logits(
        mask_logits[torch.arange(labels.shape[0], device=labels.device), labels], mask_targets
    )
    return mask_loss

def keypoints_to_heatmap(keypoints, rois, heatmap_size):
    # type: (Tensor, Tensor, int) -> Tuple[Tensor, Tensor]
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]
    scale_x = heatmap_size / (rois[:, 2] - rois[:, 0])
    scale_y = heatmap_size / (rois[:, 3] - rois[:, 1])

    offset_x = offset_x[:, None]
    offset_y = offset_y[:, None]
    scale_x = scale_x[:, None]
    scale_y = scale_y[:, None]

    x = keypoints[..., 0]
    y = keypoints[..., 1]

    x_boundary_inds = x == rois[:, 2][:, None]
    y_boundary_inds = y == rois[:, 3][:, None]

    x = (x - offset_x) * scale_x
    x = x.floor().long()
    y = (y - offset_y) * scale_y
    y = y.floor().long()

    x[x_boundary_inds] = heatmap_size - 1
    y[y_boundary_inds] = heatmap_size - 1

    valid_loc = (x >= 0) & (y >= 0) & (x < heatmap_size) & (y < heatmap_size)
    vis = keypoints[..., 2] > 0
    valid = (valid_loc & vis).long()

    lin_ind = y * heatmap_size + x
    heatmaps = lin_ind * valid

    return heatmaps, valid

def heatmaps_to_keypoints(maps, rois):
    """Extract predicted keypoint locations from heatmaps. Output has shape
    (#rois, 4, #keypoints) with the 4 rows corresponding to (x, y, logit, prob)
    for each keypoint.
    """
    # This function converts a discrete image coordinate in a HEATMAP_SIZE x
    # HEATMAP_SIZE image to a continuous keypoint coordinate. We maintain
    # consistency with keypoints_to_heatmap_labels by using the conversion from
    # Heckbert 1990: c = d + 0.5, where d is a discrete coordinate and c is a
    # continuous coordinate.
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]

    widths = rois[:, 2] - rois[:, 0]
    heights = rois[:, 3] - rois[:, 1]
    widths = widths.clamp(min=1)
    heights = heights.clamp(min=1)
    widths_ceil = widths.ceil()
    heights_ceil = heights.ceil()

    num_keypoints = maps.shape[1]

    xy_preds = torch.zeros((len(rois), 3, num_keypoints), dtype=torch.float32, device=maps.device)
    end_scores = torch.zeros((len(rois), num_keypoints), dtype=torch.float32, device=maps.device)
    for i in range(len(rois)):
        roi_map_width = int(widths_ceil[i].item())
        roi_map_height = int(heights_ceil[i].item())
        width_correction = widths[i] / roi_map_width
        height_correction = heights[i] / roi_map_height
        roi_map = F.interpolate(
            maps[i][:, None], size=(roi_map_height, roi_map_width), mode='bicubic', align_corners=False)[:, 0]
        # roi_map_probs = scores_to_probs(roi_map.copy())
        w = roi_map.shape[2]
        pos = roi_map.reshape(num_keypoints, -1).argmax(dim=1)

        x_int = pos % w
        y_int = torch.div(pos - x_int, w, rounding_mode='floor')
        # assert (roi_map_probs[k, y_int, x_int] ==
        #         roi_map_probs[k, :, :].max())
        x = (x_int.float() + 0.5) * width_correction
        y = (y_int.float() + 0.5) * height_correction
        xy_preds[i, 0, :] = x + offset_x[i]
        xy_preds[i, 1, :] = y + offset_y[i]
        xy_preds[i, 2, :] = 1
        end_scores[i, :] = roi_map[torch.arange(num_keypoints, device=roi_map.device), y_int, x_int]

    return xy_preds.permute(0, 2, 1), end_scores

def keypointrcnn_loss(keypoint_logits, proposals, gt_keypoints, keypoint_matched_idxs, keypoint3d_pred=None, keypoint3d_gt=None, mesh3d_pred=None, mesh3d_gt=None):
    # type: (Tensor, List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor]) -> Tensor
    N, K, H, W = keypoint_logits.shape
    assert H == W
    discretization_size = H
    heatmaps = []
    valid = []
    kps3d = []
    meshes3d = []

    zipped_data = zip(proposals, gt_keypoints, keypoint3d_gt, mesh3d_gt, keypoint_matched_idxs)
    for proposals_per_image, gt_kp_in_image, gt_kp3d_in_image, gt_mesh3d_in_image, midx in zipped_data:
        kp = gt_kp_in_image[midx]
        kp3d = gt_kp3d_in_image[midx]
        mesh3d = gt_mesh3d_in_image[midx]

        heatmaps_per_image, valid_per_image = keypoints_to_heatmap(kp, proposals_per_image, discretization_size)
        
        heatmaps.append(heatmaps_per_image.view(-1))
        valid.append(valid_per_image.view(-1))
        kps3d.append(kp3d.view(-1))
        meshes3d.append(mesh3d.view(-1))
        
        
    keypoint_targets = torch.cat(heatmaps, dim=0)
    valid = torch.cat(valid, dim=0).to(dtype=torch.uint8)    
    valid = torch.where(valid)[0]
    
    # torch.mean (in binary_cross_entropy_with_logits) does'nt
    # accept empty tensors, so handle it sepaartely
    
    if keypoint_targets.numel() == 0 or len(valid) == 0:
        return keypoint_logits.sum() * 0

    keypoint_logits = keypoint_logits.view(N * K, H * W)
    # TODO: understand how the tensors have different size but match in loss
    
    # Heatmap Loss
    keypoint_loss = F.cross_entropy(keypoint_logits[valid], keypoint_targets[valid])
    
    # 3D pose Loss
    keypoint_targets3d = torch.cat(kps3d, dim=0)
    keypoint3d_pred = keypoint3d_pred.view(N * K, 3)
    keypoint_targets3d = keypoint_targets3d.view(N * K, 3)
    keypoint3d_loss = F.mse_loss(keypoint3d_pred[valid], keypoint_targets3d[valid]) / 1000
    
    # 3D shape Loss
    N, K, D = mesh3d_pred.shape
    mesh_targets3d = torch.cat(meshes3d, dim=0)
    mesh3d_pred = torch.reshape(mesh3d_pred, (N * K, D))
    mesh_targets3d = torch.reshape(mesh_targets3d, (N * K, D))
    mesh3d_loss = F.mse_loss(mesh3d_pred, mesh_targets3d) / 1000

    # mesh3d_loss_smooth = calculate_smoothing_loss(mesh3d_pred[:K], K)
    # if N > 1:
    #     mesh3d_loss_smooth += calculate_smoothing_loss(mesh3d_pred[K:K*2], K)
    # mesh3d_loss += mesh3d_loss_smooth
    
    # Print the losses
    return keypoint_loss, keypoint3d_loss, mesh3d_loss

def keypointrcnn_inference(x, boxes):
    # type: (Tensor, List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
    kp_probs = []
    kp_scores = []

    boxes_per_image = [box.size(0) for box in boxes]
    x2 = x.split(boxes_per_image, dim=0)

    for xx, bb in zip(x2, boxes):
        kp_prob, scores = heatmaps_to_keypoints(xx, bb)
        kp_probs.append(kp_prob)
        kp_scores.append(scores)

    return kp_probs, kp_scores

def get_hand_object_faces(kps=778):
    src_obj = os.path.join('../HOPE/datasets/hands', 'hand_model_778.obj')
    verts, faces, aux = load_obj(src_obj)
    hand_faces = faces.verts_idx

    if kps > 778:
        src_obj = os.path.join('../HOPE/datasets/spheres', 'sphere_1000.obj')
        verts, faces, aux = load_obj(src_obj)
        object_faces = faces.verts_idx
        object_faces = object_faces + 778

        final_faces = torch.cat((hand_faces, object_faces), axis=0)
    
    else:
        final_faces = hand_faces
    return final_faces

def calculate_smoothing_loss(mesh3d, K=778):
    
    faces_idx = get_hand_object_faces(K).to('cuda:1')
            
    trg_mesh = Meshes(verts=[mesh3d], faces=[faces_idx])
            
    # src_mesh = Meshes(verts=[keypoint_targets3d[valid]], faces=[faces_idx])
    # We compare the two sets of pointclouds by computing (a) the chamfer loss
    # loss_chamfer, _ = chamfer_distance(keypoint3d_pred[valid], keypoint_targets3d[valid])
    
    # and (b) the edge length of the predicted mesh
    loss_edge = mesh_edge_loss(trg_mesh)
    # mesh normal consistency
    loss_normal = mesh_normal_consistency(trg_mesh)
    # mesh laplacian smoothing
    loss_laplacian = mesh_laplacian_smoothing(trg_mesh, method="uniform")
    # Weighted sum of the losses
    # print(loss_edge, loss_laplacian* 0.1 , loss_normal * 0.01)
    mesh3d_loss_smooth = 0.01 * loss_edge + loss_laplacian * 0.001  + loss_normal * 0.00001

    return mesh3d_loss_smooth
