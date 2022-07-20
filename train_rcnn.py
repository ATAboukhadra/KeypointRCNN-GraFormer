# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

""" import libraries"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import logging
import sys
import os

from utils.options import parse_args_function
from utils.dataset import Dataset
from utils.utils import freeze_component, calculate_keypoints
from models.keypoint_rcnn import keypointrcnn_resnet50_fpn

from h2o_utils.h2o_datapipe_pt_1_12 import create_datapipe
from h2o_utils.h2o_example_pt_1_12 import get_tar_lists, MyPreprocessor
from torch.utils.data.dataloader_experimental import DataLoader2
from h2o_utils.datapipe_helpers import collate_batch_as_dict

torch.multiprocessing.set_sharing_strategy('file_system')

args = parse_args_function()
root = args.input_file

# Define device
device = torch.device(f'cuda:{args.gpu_number[0]}' if torch.cuda.is_available() else 'cpu')

use_cuda = False
if torch.cuda.is_available():
    use_cuda = True

init_num_kps, num_keypoints = calculate_keypoints(args.object)

""" Configure a log """

log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
fh = logging.FileHandler(os.path.join(args.output_file[:-6], 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


""" load transformation functions"""

transform = transforms.Compose([transforms.ToTensor()])

""" Load Tar Files """

# Train
data_dir = '/ds-av/public_datasets/h2o/wds/shards'
input_tar_lists = get_tar_lists(data_dir, ['rgb'], subjects=[1, 2], cameras=[4])
input_tar_lists.extend(get_tar_lists(data_dir, ['rgb'], subjects=[3], scenes=['h1', 'h2', 'k1'], cameras=[4]))

annotation_tar_files = get_tar_lists(data_dir, ['annotations'], subjects=[1, 2], cameras=[4])[0]
annotation_tar_files.extend(get_tar_lists(data_dir, ['annotations'], subjects=[3], scenes=['h1', 'h2', 'k1'], cameras=[4])[0])

annotation_components = ['cam_pose', 'hand_pose', 'hand_pose_mano', 'obj_pose', 'obj_pose_rt', 'action_label', 'verb_label']

""" load model """

model = keypointrcnn_resnet50_fpn(init_num_kps=init_num_kps, num_keypoints=num_keypoints, num_classes=2, 
                                # rpn_batch_size_per_image=1,
                                # box_detections_per_img=1,
                                rpn_post_nms_top_n_train=1, rpn_post_nms_top_n_test=1, 
                                rpn_pre_nms_top_n_train=1,
                                device=device, num_features=args.num_features)
print('Keypoint RCNN is loaded')
print(model)

if torch.cuda.is_available():
    model = model.cuda(args.gpu_number[0])
    model = nn.DataParallel(model, device_ids=args.gpu_number)

    # freeze_component(model.module.backbone)
    # freeze_component(model.module.rpn)
""" load saved model"""

if args.pretrained_model != '':
    model.load_state_dict(torch.load(args.pretrained_model, map_location=f'cuda:{args.gpu_number[0]}'))
    losses = np.load(args.pretrained_model[:-4] + '-losses.npy').tolist()
    start = len(losses)
else:
    losses = []
    start = 0

"""define optimizer"""

criterion = nn.MSELoss()
# criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_step_gamma)
scheduler.last_epoch = start

keys = ['boxes', 'labels', 'keypoints', 'keypoints3d', 'mesh3d']

""" training """

if args.train:
    logging.info('Begin training the network...')
    
    for epoch in range(start, args.num_iterations):  # loop over the dataset multiple times
        
        train_loss2d = 0.0
        running_loss2d = 0.0
        running_loss3d = 0.0
        running_mesh_loss3d = 0.0
        

        """ load datasets """
        shuffle_buffer_size = 100
        datapipe = create_datapipe(input_tar_lists, annotation_tar_files, annotation_components, shuffle_buffer_size)
        datapipe = datapipe.map(fn=MyPreprocessor('../mano_v1_2/models/',
                                                  '../datasets/objects/mesh_1000/',
                                                  '/ds-av/public_datasets/h2o/wds/'))
        trainloader = DataLoader2(datapipe, batch_size=2, num_workers=2, collate_fn=collate_batch_as_dict, pin_memory=True, parallelism_mode='mp')
        
        for i, tr_data in enumerate(trainloader):
            
            # get the inputs
            data_dict = tr_data
         
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward
            targets = [{k: v.to(device) for k, v in t.items() if k in keys} for t in data_dict]
            inputs = [t['inputs'].to(device) for t in data_dict]
            loss_dict = model(inputs, targets)

            # Calculate Loss
            loss = sum(loss for loss in loss_dict.values())
            
            # Backpropagate
            loss.backward()
            optimizer.step()

            # print statistics
            train_loss2d += loss_dict['loss_keypoint'].data
            running_loss2d += loss_dict['loss_keypoint'].data
            running_loss3d += loss_dict['loss_keypoint3d'].data
            running_mesh_loss3d += loss_dict['loss_mesh3d'].data
            
            if (i+1) % args.log_batch == 0:    # print every log_iter mini-batches
                logging.info('[%d, %5d] loss 2d: %.5f, loss 3d: %.5f, mesh loss 3d:%.5f' % 
                (epoch + 1, i + 1, running_loss2d / args.log_batch, running_loss3d / args.log_batch, running_mesh_loss3d / args.log_batch))
                
                running_mesh_loss3d = 0.0
                running_loss2d = 0.0
                running_loss3d = 0.0
        losses.append((train_loss2d / (i+1)).cpu().numpy())
        
        if (epoch+1) % args.snapshot_epoch == 0:
            torch.save(model.state_dict(), args.output_file+str(epoch+1)+'.pkl')
            np.save(args.output_file+str(epoch+1)+'-losses.npy', np.array(losses))

        if args.val and (epoch+1) % args.val_epoch == 0:
            val_loss2d = 0.0
            val_loss3d = 0.0
            val_mesh_loss3d = 0.0
            for v, val_data in enumerate(valloader):
                # get the inputs
                data_dict = val_data
            
                # wrap them in Variable
                targets = [{k: v.to(device) for k, v in t[0].items() if k in keys} for t in data_dict]
                inputs = [t[0]['inputs'].to(device) for t in data_dict]    
                loss_dict = model(inputs, targets)
                
                val_loss2d += loss_dict['loss_keypoint'].data
                val_loss3d += loss_dict['loss_keypoint3d'].data
                val_mesh_loss3d += loss_dict['loss_mesh3d'].data
            
            logging.info('val loss 2d: %.5f, val loss 3d: %.5f, val mesh loss 3d: %.5f' % (val_loss2d / (v+1), val_loss3d / (v+1), val_mesh_loss3d / (v+1)))
        
        if args.freeze and epoch == 0:
            logging.info('Freezing Backbone and RPN and RoI heads ..')            
            freeze_component(model.module.backbone)
            freeze_component(model.module.rpn)
            # Box models
            freeze_component(model.module.roi_heads.box_roi_pool)
            freeze_component(model.module.roi_heads.box_head)
            freeze_component(model.module.roi_heads.box_predictor)
            # Keypoint models
            freeze_component(model.module.roi_heads.keypoint_roi_pool)
            freeze_component(model.module.roi_heads.keypoint_head)
            freeze_component(model.module.roi_heads.keypoint_predictor)

        # Decay Learning Rate
        scheduler.step()
    
    logging.info('Finished Training')