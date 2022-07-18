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

torch.multiprocessing.set_sharing_strategy('file_system')

def collate_fn(batch):
    return tuple(zip(batch))

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

""" load datasets """

if args.train:
    trainset = Dataset(root=root, load_set='train', transform=transform, num_keypoints=num_keypoints)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=32, collate_fn=collate_fn)    
    logging.info('Train files loaded')
    logging.info(f'size of training set: {len(trainset)}')

if args.val:
    valset = Dataset(root=root, load_set='val', transform=transform, num_keypoints=num_keypoints)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=16, collate_fn=collate_fn)
    logging.info('Validation files loaded')
    logging.info(f'size of validation set: {len(valset)}')

""" load model """

model = keypointrcnn_resnet50_fpn(init_num_kps=init_num_kps, num_keypoints=num_keypoints, num_classes=2, 
                                # rpn_batch_size_per_image=1,
                                box_detections_per_img=1,
                                rpn_post_nms_top_n_train=1, rpn_post_nms_top_n_test=1, 
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
        
        for i, tr_data in enumerate(trainloader):
            
            # get the inputs
            data_dict = tr_data
         
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward
            targets = [{k: v.to(device) for k, v in t[0].items() if k in keys} for t in data_dict]
            inputs = [t[0]['inputs'].to(device) for t in data_dict]
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
            logging.info('Freezing Backbone and RPN ..')
            freeze_component(model.module.backbone)
            freeze_component(model.module.rpn)

        # Decay Learning Rate
        scheduler.step()
    
    logging.info('Finished Training')