import os
import torch
from itertools import product
from typing import List, Optional

from torch.utils.data.dataloader_experimental import DataLoader2
from torch.utils.data.datapipes.datapipe import DataChunk

from .datapipe_helpers import collate_batch_as_tuple
from .h2o_datapipe_pt_1_12 import create_datapipe


from .preprocessing_functions import Preprocessor, create_rcnn_data, index_containing_substring
from .visualization import visualize


class MyPreprocessor:
    def __init__(self, mano_model_path: str, objects_path: str, ds_dir: str):
        self.preprocessor = Preprocessor(mano_model_path, objects_path, ds_dir)

    def __call__(self, sample: DataChunk):
        """ This is an example of how to preprocess the annotations and create a new tuple.

        :param sample: The sample that shall be preprocessed.
        :return: The new sample.
        """

        # Get the index of the required data
        filenames = [s[0] for s in sample]
        data = ['rgb', 'hand_pose', 'mano', 'obj_pose', 'obj_pose_rt']
        data_idx = [index_containing_substring(filenames, s) for s in data]
        data_idx_dict = dict(zip(data, data_idx))

        # RGB image is the first element in the tuple
        rgb_path = sample[data_idx_dict['rgb']][0]
        rgb_image = sample[data_idx_dict['rgb']][1]

        # Subject / Scene / object / cam key for the sample
        key = sample[0][0].split('/')[-1].split('.')[0][:-7]

        # Hand pose
        hand_pose3d = sample[data_idx_dict['hand_pose']][1]

        # Decoding Mano Params
        mano_params = sample[data_idx_dict['mano']][1]
        left_hand_verts = self.preprocessor.decode_mano(mano_params[:62], is_right=False)
        right_hand_verts = self.preprocessor.decode_mano(mano_params[62:], is_right=True)

        # Object label is stored at the first index of the last row of the object pose
        object_pose = sample[data_idx_dict['obj_pose']][1]
        object_label = int(object_pose[-1][0])
        # Load Object Mesh
        object_verts = self.preprocessor.load_object_mesh(object_label)

        # Project object vertices to camera coordinates
        obj_pose_rt = sample[data_idx_dict['obj_pose_rt']][1]
        homogeneous_ones = torch.ones((object_verts.shape[0], 1))
        object_verts = torch.cat((object_verts, homogeneous_ones), axis=1)
        object_verts = obj_pose_rt.matmul(object_verts.T).T[:, :-1] * 1000

        # Append hand meshes with object mesh
        mesh3d = torch.cat((left_hand_verts, right_hand_verts, object_verts), axis=0)

        # Append hand poses with object pose
        pose3d = torch.cat((hand_pose3d, object_pose[:-1]), axis=0)

        # Project 3D pose and 3D shape into 2D using camera intrinsics
        mesh2d = self.preprocessor.project_3D_points(mesh3d, key)
        pose2d = self.preprocessor.project_3D_points(pose3d, key)

        # Create bounding boxes and keypoints to train Keypoint RCNN
        boxes, labels, keypoints = create_rcnn_data(pose2d, merge_objects=True)

        # Return the new sample as a dict
        new_sample = (rgb_path, rgb_image, pose2d, pose3d, mesh2d, mesh3d, boxes, labels, keypoints)

        output_dict = {
            'rgb_path': rgb_path,
            'rgb_image': rgb_image,
            'pose2d': pose2d,
            'pose3d': pose3d,
            'mesh2d': mesh2d,
            'mesh3d': mesh3d,
            'boxes': boxes,
            'labels': labels,
            'keypoints': keypoints
        }

        return new_sample


def my_augmentation_fn(sample: DataChunk):
    """ This is an example of how to include data augmentation functions in the pipeline.

    :param sample: The sample that shall be augmented.
    :return: The augmented sample.
    """
    # Add your augmentation code here.
    return sample


def get_tar_lists(base_path: str, data_components: List[str], subjects: Optional[List[int]] = None,
                  scenes: Optional[List[str]] = None, sequences: Optional[List[int]] = None,
                  cameras: Optional[List[int]] = None):
    """ Create for each requested component of the dataset a sequence of tar file shards that include the specified
    subjects, scenes, objects and cameras.

    :param base_path: The base path where the shards are stored.
    :param data_components: A list of the components of the datasets that shall be used. The valid options are:
       'rgb', 'rgb256', 'depth', 'annotations'.
    :param subjects: A list of the subjects that shall be included. The valid options are: 1, 2, 3, 4.
    :param scenes: A list of the scenes that shall be included. The options are: 'h1', 'h2', 'k1', 'k2', 'o1', 'o2'.
    :param sequences: A list of the sequences that shall be included. The options are: 0, 1, 2, 3, 4, 5, 6, 7.
    :param cameras: The cameras that shall be included. The options are: 0, 1, 2, 3, 4.
    :return: For each requested component of the dataset a list of tar files. (Returned as a list of lists.)
    """
    if subjects is None:
        subjects = [1, 2, 3, 4]
    if scenes is None:
        scenes = ['h1', 'h2', 'k1', 'k2', 'o1', 'o2']
    if sequences is None:
        sequences = [0, 1, 2, 3, 4, 5, 6, 7]
    if cameras is None:
        cameras = [0, 1, 2, 3, 4]
    tar_file_lists = []
    # Create tar list. Skip combinations that do not exist.
    for data_component in data_components:
        tar_file_list = [os.path.join(base_path, f'subject{subject}_{scene}_{obj}_cam{camera}_{data_component}.tar')
                         for subject, scene, obj, camera
                         in product(subjects, scenes, sequences, cameras)
                         if (data_component != 'rgb256' or (data_component == 'rgb256' and camera == 4)) and
                         not (obj == 7 and ((subject == 1 and scene == "k2") or
                                            (subject == 2 and scene in ["h2", "k1", "k2", "o1", "o2"]) or
                                            (subject == 4 and scene in ["k1", "o2"])))]
        tar_file_lists.append(sorted(tar_file_list))
    return tar_file_lists


def train_example(data_dir: str):
    """ Example of how to use the data streaming pipeline of the H2O dataset for training.

    :param data_dir: The path to the directory containing the H2O dataset.
    """
    print("Running training example.")
    shuffle_buffer_size = 1000
    # Generate list of tar files to use for training
    # data_components = ['rgb', 'rgb256', 'depth', 'annotations']
    input_tar_lists = get_tar_lists(data_dir, ['rgb'], subjects=[1, 2, 3], cameras=[4])
    annotation_tar_files = get_tar_lists(data_dir, ['annotations'], subjects=[1, 2, 3], cameras=[4])[0]
    annotation_components = ['cam_pose', 'hand_pose', 'hand_pose_mano', 'obj_pose', 'obj_pose_rt', 'action_label',
                             'verb_label']
    # How many epochs to train for
    epochs = 1
    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch}:")
        datapipe = create_datapipe(input_tar_lists, annotation_tar_files, annotation_components, shuffle_buffer_size)
        datapipe = datapipe.map(fn=MyPreprocessor('../mano_v1_2/models/',
                                                  '../datasets/objects/mesh_1000/',
                                                  '/ds-av/public_datasets/h2o/wds/'))
        datapipe = datapipe.map(fn=my_augmentation_fn)
        train_loader = DataLoader2(datapipe,
                                   batch_size=2,
                                   num_workers=2,
                                   collate_fn=collate_batch_as_tuple,
                                   pin_memory=True,
                                   parallelism_mode='mp')
        # Iterate over the batches. The data loader returns the file names and the batch data.
        for file_names, batch in train_loader:
            print("The batch contains data from the following files:")
            print(file_names)
            # visualize(batch)
            # This is just an example, so break after the first batch.
            break
    print("Done.")


def val_example(data_dir: str):
    """ Example of how to use the data streaming pipeline of the H2O dataset for testing.

    :param data_dir: The path to the directory containing the H2O dataset.
    """
    print("Running validation example.")
    shuffle_buffer_size = 100
    # Generate list of tar files to use for validation
    input_tar_lists = get_tar_lists(data_dir, ['rgb'], subjects=[4], cameras=[4])
    annotation_tar_files = get_tar_lists(data_dir, ['annotations'], subjects=[4], cameras=[4])[0]
    # List of the annotation components that shall be kept
    annotation_components = ['cam_pose', 'hand_pose', 'hand_pose_mano', 'obj_pose', 'obj_pose_rt', 'action_label',
                             'verb_label']
    # Create an instance of the dataset. Shards do not need to be shuffled for validation.
    datapipe = create_datapipe(input_tar_lists, annotation_tar_files, annotation_components, shuffle_buffer_size,
                               shuffle_shards=False)
    datapipe = datapipe.map(fn=MyPreprocessor('../mano_v1_2/models/',
                                              '../datasets/objects/mesh_1000/',
                                              '/ds-av/public_datasets/h2o/wds/'))
    # Comments on the parameters of DataLoader 2:
    # shuffle: Save time and memory by turning off shuffling for validation.
    # num_workers: Higher values increase preprocessing speed, memory consumption and the initial delay before the
    # training starts caused be loading the first shard(s) for each worker.
    # collate_fn: This function is used to assemble a batch from the individual samples. It defines the structure of
    #  the data returned by the data loader.
    # pin_memory: Place final tensors in pinned memory so transfer to GPU can be done more quickly.
    # parallelism_mode: Allows to choose the parallelism mode used if num_workers > 0.
    #   'thread' makes the debugger work with workers > 0.
    #   'mp' makes pin_memory work with workers > 0.
    val_loader = DataLoader2(datapipe,
                             batch_size=1,
                             shuffle=False,
                             num_workers=2,
                             collate_fn=collate_batch_as_tuple,
                             pin_memory=True,
                             parallelism_mode='mp')
    for inputs, targets in val_loader:
        print("The batch contains the following inputs:")
        print(inputs)
        print(targets)
        # This is just an example, so break after the first batch.
        break
    print("Done.")


if __name__ == '__main__':
    dataset_dir = "/ds-av/public_datasets/h2o/wds/shards"
    train_example(dataset_dir)
    val_example(dataset_dir)
