# -*- coding: utf-8 -*-

# import libraries
import numpy as np
import os
import torch.utils.data as data
import cv2
import os.path
import io
import torch 
from PIL import Image
from utils.train_utils import calculate_bounding_box, create_rcnn_data


class Dataset(data.Dataset):
    """# Dataset Class """

    def __init__(self, root='./', load_set='train', transform=None, return_mesh=False, object=False,  hdf5_file=None):

        self.root = root
        self.transform = transform
        self.return_mesh = return_mesh
        self.object = object
        self.hdf5 = hdf5_file

        # TODO: add depth transformation
        self.load_set = load_set  # 'train','val','test'
        self.images = np.load(os.path.join(root, 'images-%s.npy' % self.load_set))
        self.points2d = np.load(os.path.join(root, 'points2d-%s.npy' % self.load_set))
        self.points3d = np.load(os.path.join(root, 'points3d-%s.npy' % self.load_set))

        self.mesh2d = np.load(os.path.join(root, 'mesh2d-%s.npy' % self.load_set))
        if self.return_mesh:
            self.mesh = np.load(os.path.join(root, 'mesh3d-%s.npy' % self.load_set))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, points2D, points3D).
        """

        image_path = self.images[index]

        point2d = self.points2d[index]
        point3d = self.points3d[index]

        # Loading Hand Mesh for hand bounding box
        mesh2d = self.mesh2d[index][:778]
                
        # Load image and apply preprocessing if any
        if self.hdf5 is not None:
            data = np.array(self.hdf5[image_path])
            original_image = np.array(Image.open(io.BytesIO(data)))[..., :3]
        else:
            original_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        inputs = self.transform(original_image)  # [:3]

        if self.load_set != 'test':
            bb_hand = calculate_bounding_box(mesh2d, increase=True)
            bb_object = calculate_bounding_box(point2d[21:])
            if self.return_mesh:
                if self.object:
                    # Load complete hand-object mesh
                    mesh2d = self.mesh2d[index]
            if self.return_mesh:
                boxes, labels, keypoints, keypoints3d = create_rcnn_data(bb_hand, bb_object, mesh2d, pose=False, obj=self.object)
            else:
                boxes, labels, keypoints, keypoints3d = create_rcnn_data(bb_hand, bb_object, point2d, point3d, pose=True, obj=self.object)

        else:
            bb_hand, bb_object, mesh2d = np.array([]), np.array([]), np.array([])
            boxes, labels, keypoints = torch.Tensor([]), torch.Tensor([]), torch.Tensor([])

        data = {
            'path': image_path,
            'original_image': original_image,
            'inputs': inputs,
            'point2d': point2d,
            'point3d': point3d,
            'mesh2d': mesh2d,
            'bb_hand': bb_hand,
            'bb_object': bb_object,
            'boxes': boxes,
            'labels': labels,
            'keypoints': keypoints,
            'keypoints3d': keypoints3d 
        }

        return data

    def __len__(self):
        return len(self.images)
