import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'train'))
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ipdb
from torch.nn import init
from model_util_old import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, NUM_OBJECT_POINT
from model_util_old import point_cloud_masking, parse_output_to_tensors
from model_util_old import FrustumPointNetLoss
#from model_util import g_type2class, g_class2type, g_type2onehotclass
#from model_util import g_type_mean_size
#from model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER
from provider import compute_box3d_iou

from easydict import EasyDict
from backbones_3d.generalpointnet2_backbone import GeneralPointNet2MSG


NUM_HEADING_BIN = 12
NUM_SIZE_CLUSTER = 8 # one cluster for each type
NUM_OBJECT_POINT = 512

g_type2class={'Car':0, 'Van':1, 'Truck':2, 'Pedestrian':3,
              'Person_sitting':4, 'Cyclist':5, 'Tram':6, 'Misc':7}
g_class2type = {g_type2class[t]:t for t in g_type2class}
g_type2onehotclass = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}

g_type_mean_size = {'Car': np.array([3.88311640418,1.62856739989,1.52563191462]),
                    'Van': np.array([5.06763659,1.9007158,2.20532825]),
                    'Truck': np.array([10.13586957,2.58549199,3.2520595]),
                    'Pedestrian': np.array([0.84422524,0.66068622,1.76255119]),
                    'Person_sitting': np.array([0.80057803,0.5983815,1.27450867]),
                    'Cyclist': np.array([1.76282397,0.59706367,1.73698127]),
                    'Tram': np.array([16.17150617,2.53246914,3.53079012]),
                    'Misc': np.array([3.64300781,1.54298177,1.92320313])}


g_mean_size_arr = np.zeros((NUM_SIZE_CLUSTER, 3)) # size clustrs
for i in range(NUM_SIZE_CLUSTER):
    g_mean_size_arr[i,:] = g_type_mean_size[g_class2type[i]]


class PointNetEstimation(nn.Module):
    def __init__(self,n_classes=3):
        super(PointNetEstimation, self).__init__()
        self.config = {'NAME': 'GeneralPointNet2MSG', 'ENCODER': [
            {'samplers': [{'name': 'd-fps', 'sample': 128}],
             'groupers': [{'name': 'ball', 'query': {'radius': 0.2, 'neighbour': 16}, 'mlps': [16, 16, 32]},
                          {'name': 'ball', 'query': {'radius': 0.4, 'neighbour': 32}, 'mlps': [32, 32, 64]}],
             'aggregation': {'name': 'cat-mlps', 'mlps': [64]}},
            {'samplers': [{'name': 'd-fps', 'sample': 64}],
             'groupers': [{'name': 'ball', 'query': {'radius': 0.2, 'neighbour': 16}, 'mlps': [64, 64, 128]},
                          {'name': 'ball', 'query': {'radius': 0.4, 'neighbour': 32}, 'mlps': [64, 96, 128]}],
             'aggregation': {'name': 'cat-mlps', 'mlps': [128]}},
            {'samplers': [{'name': 'd-fps', 'sample': 1}],
             'groupers': [{'name': 'ball', 'query': {'radius': 0.2, 'neighbour': 16}, 'mlps': [128, 128, 256]},
                          {'name': 'ball', 'query': {'radius': 0.4, 'neighbour': 32}, 'mlps': [128, 256, 256]}],
             'aggregation': {'name': 'cat-mlps', 'mlps': [256]}}]}

        self.config = EasyDict(self.config)
        self.backbone = GeneralPointNet2MSG(self.config, input_channels=3)

        self.fc1 = nn.Linear(256 + n_classes, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, 3 + NUM_HEADING_BIN * 2 + NUM_SIZE_CLUSTER * 4)

    def forward(self, pts, one_hot_vec):
        bs = pts.size()[0]
        n_pts = pts.size()[2]

        # (B, N, 3) tensor of the xyz coordinates of the features
        pts = pts.permute(0, 2, 1)

        pts, features = self._break_up_pc(pts)

        batch_dict = {'points': pts}
        # print(pts.shape)
        features = self.backbone(batch_dict)
        l3_points = features['point_features']
        global_feat = l3_points.view(bs, -1)

        expand_one_hot_vec = one_hot_vec.view(bs, -1)  # bs,3
        expand_global_feat = torch.cat([global_feat, expand_one_hot_vec], 1)  # bs,515

        x = F.relu(self.bn1(self.fc1(expand_global_feat)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        box_pred = self.fc3(x)

        return box_pred

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None)

        return xyz, features

class STNxyz(nn.Module):
    def __init__(self,n_classes=3):
        super(STNxyz, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        #self.conv4 = torch.nn.Conv1d(256, 512, 1)
        self.fc1 = nn.Linear(256+n_classes, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 3)

        init.zeros_(self.fc3.weight)
        init.zeros_(self.fc3.bias)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.fcbn1 = nn.BatchNorm1d(256)
        self.fcbn2 = nn.BatchNorm1d(128)
    def forward(self, pts,one_hot_vec):
        bs = pts.shape[0]
        x = F.relu(self.bn1(self.conv1(pts)))# bs,128,n
        x = F.relu(self.bn2(self.conv2(x)))# bs,128,n
        x = F.relu(self.bn3(self.conv3(x)))# bs,256,n
        x = torch.max(x, 2)[0]# bs,256
        expand_one_hot_vec = one_hot_vec.view(bs, -1)# bs,3
        x = torch.cat([x, expand_one_hot_vec],1)#bs,259
        x = F.relu(self.fcbn1(self.fc1(x)))# bs,256
        x = F.relu(self.fcbn2(self.fc2(x)))# bs,128
        x = self.fc3(x)# bs,
        ###if np.isnan(x.cpu().detach().numpy()).any():
        ###    ipdb.set_trace()
        return x

class FrustumPointNetv2(nn.Module):
    def __init__(self,n_classes=4,n_channel=3):
        super(FrustumPointNetv2, self).__init__()
        self.n_classes = n_classes
        self.STN = STNxyz(n_classes=4)
        self.est = PointNetEstimation(n_classes=4)

    def forward(self, pts, one_hot_vec):#bs,4,n
        # 3D Instance Segmentation PointNet
        # logits = self.InsSeg(pts,one_hot_vec)#bs,n,2
        # print('logits:',logits.shape,logits.dtype)


        # Mask Point Centroid
        # logits = torch.zeros(size=(pts.shape[0],1024,2),dtype=torch.float32).cuda()
        # logits[:, :, :] = 0
        # logits[:, :, :] = 1

        # print('pts:',pts.shape,pts.dtype)
        # print('one_hot_vec:',one_hot_vec.shape,one_hot_vec.dtype)
        #
        # # All logits correct
        # # logits = bs,n,2
        #
        # # Mask Point Centroid
        # object_pts_xyz, mask_xyz_mean, mask = \
        #          point_cloud_masking(pts, logits)###logits.detach()

        # print('pts:',pts.shape,pts.dtype)
        pts = pts[:, :3, :]
        # print(one_hot_vec)

        clusters_mean = torch.mean(pts, 2).cuda()
        reshaped_center_delta = clusters_mean.view(clusters_mean.shape[0], -1, 1)
        repeated_center_delta = reshaped_center_delta.repeat(1, 1, pts.shape[-1])
        object_pts_xyz = pts - repeated_center_delta



        # T-Net
        object_pts_xyz = object_pts_xyz.cuda()
        center_delta = self.STN(object_pts_xyz,one_hot_vec)#(32,3)
        stage1_center = center_delta + clusters_mean#(32,3)

        if(np.isnan(stage1_center.cpu().detach().numpy()).any()):
            ipdb.set_trace()
        object_pts_xyz_new = object_pts_xyz - \
                    center_delta.view(center_delta.shape[0],-1,1).repeat(1,1,object_pts_xyz.shape[-1])

        # 3D Box Estimation
        box_pred = self.est(object_pts_xyz_new,one_hot_vec)#(32, 59)

        #dummy logits
        logits = 0
        mask = 0

        center_boxnet, \
        heading_scores, heading_residuals_normalized, heading_residuals, \
        size_scores, size_residuals_normalized, size_residuals = \
                parse_output_to_tensors(box_pred, logits, mask, stage1_center)

        center = center_boxnet + stage1_center #bs,3
        return logits, mask, stage1_center, center_boxnet, \
            heading_scores, heading_residuals_normalized, heading_residuals, \
            size_scores, size_residuals_normalized, size_residuals, center




if __name__ == '__main__':
    #python models/pointnet.py
    points = torch.zeros(size=(32,4,1024),dtype=torch.float32)
    label = torch.ones(size=(32,3))
    model = FrustumPointNetv2()
    logits, mask, stage1_center, center_boxnet, \
            heading_scores, heading_residuals_normalized, heading_residuals, \
            size_scores, size_residuals_normalized, size_residuals, center \
            = model(points,label)
    print('logits:',logits.shape,logits.dtype)
    print('mask:',mask.shape,mask.dtype)
    print('stage1_center:',stage1_center.shape,stage1_center.dtype)
    print('center_boxnet:',center_boxnet.shape,center_boxnet.dtype)
    print('heading_scores:',heading_scores.shape,heading_scores.dtype)
    print('heading_residuals_normalized:',heading_residuals_normalized.shape,\
          heading_residuals_normalized.dtype)
    print('heading_residuals:',heading_residuals.shape,\
          heading_residuals.dtype)
    print('size_scores:',size_scores.shape,size_scores.dtype)
    print('size_residuals_normalized:',size_residuals_normalized.shape,\
          size_residuals_normalized.dtype)
    print('size_residuals:',size_residuals.shape,size_residuals.dtype)
    print('center:', center.shape,center.dtype)
    '''
    logits: torch.Size([32, 1024, 2]) torch.float32
    mask: torch.Size([32, 1024]) torch.float32
    stage1_center: torch.Size([32, 3]) torch.float32
    center_boxnet: torch.Size([32, 3]) torch.float32
    heading_scores: torch.Size([32, 12]) torch.float32
    heading_residual_normalized: torch.Size([32, 12]) torch.float32
    heading_residual: torch.Size([32, 12]) torch.float32
    size_scores: torch.Size([32, 8]) torch.float32
    size_residuals_normalized: torch.Size([32, 8, 3]) torch.float32
    size_residuals: torch.Size([32, 8, 3]) torch.float32
    center: torch.Size([32, 3]) torch.float32
    '''
    loss = FrustumPointNetLoss()
    mask_label = torch.zeros(32,1024).float()
    center_label = torch.zeros(32,3).float()
    heading_class_label = torch.zeros(32).long()
    heading_residuals_label = torch.zeros(32).float()
    size_class_label = torch.zeros(32).long()
    size_residuals_label = torch.zeros(32,3).float()
    output_loss = loss(logits, mask_label, \
                center, center_label, stage1_center, \
                heading_scores, heading_residuals_normalized, heading_residuals, \
                heading_class_label, heading_residuals_label, \
                size_scores,size_residuals_normalized,size_residuals,
                size_class_label,size_residuals_label)
    print('output_loss',output_loss)
    print()