import torch
import numpy as np

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


def create_rcnn_data(bb_hand, bb_object, point2d, point3d, pose=True, obj=True):
    ''' Prepares data for an RCNN by creating tensors for Bounding boxes, labels and keypoints with their visibility'''

    # Boxes and Labels
    if obj:
        boxes = torch.Tensor(np.vstack((bb_hand, bb_object))).float()
        labels = torch.from_numpy(np.array([1, 2])) # 1 for hand and 2 for object
    else:
        boxes = torch.Tensor(bb_hand[np.newaxis, ...]).float()
        labels = torch.from_numpy(np.array([1])) # 1 for hand and 2 for object
    
    # Appending visibility TODO: change this to actual visibility
    visibility = np.ones(point2d.shape[0]).reshape(-1, 1)
    keypoints = np.append(point2d, visibility, axis=1)

    if pose: # In case of Pose where hand has 21 vertices and object has 8
        hand_keypoints = keypoints[:21]
        hand_keypoints3d = point3d[:21]

        if obj:
            obj_keypoints = keypoints[21:]
            # Append dummy points to the objects keypoints to match the number of hands keypoints
            hand_obj_diff = hand_keypoints.shape[0] - obj_keypoints.shape[0]
            dummy_points = np.zeros((hand_obj_diff, 3))
            obj_keypoints = np.append(obj_keypoints, dummy_points, 0)

    else: # In case of Shape where hand has 778 vertices and object has >= 1000
        hand_keypoints = keypoints[:778]
        hand_keypoints3d = point3d[:778]

        if obj:
            obj_keypoints = keypoints[778:]
            # Append dummy points to the hand's keypoints to match the number of object's keypoints
            hand_obj_diff = obj_keypoints.shape[0] - hand_keypoints.shape[0] 
            dummy_points = np.zeros((hand_obj_diff, 3))
            hand_keypoints = np.append(hand_keypoints, dummy_points, 0)
    
    # Append keypoints
    if obj:
        final_keypoints = torch.Tensor(np.stack((hand_keypoints, obj_keypoints))).float()
    else:
        final_keypoints = torch.Tensor(hand_keypoints[np.newaxis, ...]).float()
        final_keypoints3d = torch.Tensor(hand_keypoints3d[np.newaxis, ...]).float()


    return boxes, labels, final_keypoints, final_keypoints3d