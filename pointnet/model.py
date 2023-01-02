from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


"""
TNet用来进行旋转操作
T1's k = 3, and T2's k = 64
"""
class TNetkd(nn.Module):
    def __init__(self, k):
        super(TNetkd, self).__init__()
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)

        self.bn1 = nn.BarchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm3d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0] # batch_size * k * n_pts
        x = F.relu(self.bn1(self.conv1(x))) 
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x))) # batch_size * 1024 * n_pts
        x = torch.max(x, 2, keepdim=True)[0] # batch_size * 1024 * 1
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x) # batch_size * k^2 * 1

        # 这么复杂的一段代码只是为了获得一个单位矩阵 I
        iden = Variable(torch.from_numpy(
            np.eye(self.k).flatten().astype(np.float32)
        )).view(1, self.k * self.k).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k) # batch_size * k * k
        return x


"""
PointNet使用MLP提取局部特征和全局特征的模块
"""
class PointNetFeature(nn.Module):
    def __init__(self, global_feature = True, feature_transform = False):
        super(PointNetFeature, self).__init__()
        self.t1 = TNetkd(k = 3)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feature = global_feature
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.t2 = TNetkd(k = 64)

    def forward(self, x):
        n_pts = x.size()[2] # x为原始数据，batch_size * 3 * points
        trans = self.t1(x) # batch_size * 3 * 3
        x = x.transpose(2, 1) # batch_size * points * 3
        x = torch.bmm(x, trans) # batch_size * points * 3
        x = x.transpose(2, 1) # batch_size * 3 * points
        x = F.relu(self.bn1(self.conv1(x))) # batch_size * 64 * points

        if self.feature_transform:
            trans_feature = self.t2(x)
            x = x.transpose(2, 1) # 
            x = torch.bmm(x, trans_feature)
            x = x.transpose(2, 1)
        else:
            trans_feature = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x)) # batch_size * 1024 * points
        x = torch.max(x, 2, keepdim=True)[0] # batch_size * 1024 * 1
        x = x.view(-1, 1024) # batch_size * 1024
        if self.global_feature:
            return x, trans, trans_feature
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feature


"""
PointNet应用于segmentation的新增模块
"""
class PointNetSemanticSegmentation(nn.Module):
    def __init__(self, k = 2, feature_transform = False):
        super(PointNetSemanticSegmentation, self).__init__()
        self.k = k
        self.feature_transform = feature_transform
        self.feature = PointNetFeature(global_feat = False, feature_transform = feature_transform)
        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feature = self.feature(x) # batch_size * 1088 * n_pts
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn2(self.conv3(x)))
        x = self.conv4(x) # batch_size * k * n_pts
        x = x.transpose(2,1).contiguous() # batch_size * n_pts * k
        x = F.log_softmax(x.view(-1, self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans, trans_feature