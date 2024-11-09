"""
Code adapted from https://github.com/fxia22/pointnet.pytorch/blob/master/pointnet/model.py.
"""

from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.num_params = sum([p.numel() for p in self.parameters()])

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = (
            Variable(
                torch.from_numpy(
                    np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)
                )
            )
            .view(1, 9)
            .repeat(batchsize, 1)
        )
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64, hidden_1=1024, hidden_2=512, hidden_3=256):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, hidden_1, 1)
        self.fc1 = nn.Linear(hidden_1, hidden_2)
        self.fc2 = nn.Linear(hidden_2, hidden_3)
        self.fc3 = nn.Linear(hidden_3, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(hidden_1)
        self.bn4 = nn.BatchNorm1d(hidden_2)
        self.bn5 = nn.BatchNorm1d(hidden_3)
        self.hidden_1 = hidden_1

        self.k = k

        self.num_params = sum([p.numel() for p in self.parameters()])

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.hidden_1)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = (
            Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32)))
            .view(1, self.k * self.k)
            .repeat(batchsize, 1)
        )
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class ShallowSTNkd(nn.Module):
    """
    Like `STNkd`, but less layers.
    """

    def __init__(self, k=64, hidden_1=512, hidden_2=256):
        super(ShallowSTNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, hidden_1, 1)
        self.fc1 = nn.Linear(hidden_1, hidden_2)
        self.fc2 = nn.Linear(hidden_2, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(hidden_1)
        self.bn3 = nn.BatchNorm1d(hidden_2)
        self.hidden_1 = hidden_1

        self.k = k

        self.num_params = sum([p.numel() for p in self.parameters()])

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.hidden_1)

        x = F.relu(self.bn3(self.fc1(x)))
        x = self.fc2(x)

        iden = (
            Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32)))
            .view(1, self.k * self.k)
            .repeat(batchsize, 1)
        )
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetfeat(nn.Module):
    def __init__(
        self,
        global_feat=True,
        feature_transform=False,
        d=2,
        d_out=1024,
        stn_1=1024,
        stn_2=512,
        stn_3=256,
    ):
        super(PointNetfeat, self).__init__()
        self.stn = STNkd(k=d, hidden_1=stn_1, hidden_2=stn_2, hidden_3=stn_3)
        self.conv1 = torch.nn.Conv1d(d, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, d_out, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(d_out)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64, hidden_1=stn_1, hidden_2=stn_2, hidden_3=stn_3)
        self.d_out = d_out
        self.num_params = sum([p.numel() for p in self.parameters()])

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.d_out)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, self.d_out, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class ShallowPointNetfeat(nn.Module):
    """
    Like `PointNetfeat`, but less layers.
    """

    def __init__(
        self,
        global_feat=True,
        feature_transform=False,
        d=2,
        d_out=512,
        stn_1=512,
        stn_2=256,
    ):
        super(ShallowPointNetfeat, self).__init__()
        self.stn = ShallowSTNkd(k=d, hidden_1=stn_1, hidden_2=stn_2)
        self.conv1 = torch.nn.Conv1d(d, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, d_out, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(d_out)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = ShallowSTNkd(k=64, hidden_1=stn_1, hidden_2=stn_2)
        self.d_out = d_out
        self.num_params = sum([p.numel() for p in self.parameters()])

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = self.bn2(self.conv2(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.d_out)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, self.d_out, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNetCls(nn.Module):
    """
    This is the main Pointnet class for classification.
    Default has 1.6m parameters.
    :param k: number of classes.
    :param d: dimension of the points of the point cloud (e.g. 2 or 3).
    """

    def __init__(
        self,
        k=10,
        d=2,
        d_feat=1024,
        d_hidden_1=512,
        d_hidden_2=256,
        stn_1=1024,
        stn_2=512,
        stn_3=256,
        feature_transform=False,
    ):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(
            global_feat=True,
            feature_transform=feature_transform,
            d=d,
            d_out=d_feat,
            stn_1=stn_1,
            stn_2=stn_2,
            stn_3=stn_3,
        )
        self.fc1 = nn.Linear(d_feat, d_hidden_1)
        self.fc2 = nn.Linear(d_hidden_1, d_hidden_2)
        self.fc3 = nn.Linear(d_hidden_2, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(d_hidden_1)
        self.bn2 = nn.BatchNorm1d(d_hidden_2)
        self.relu = nn.ReLU()
        self.num_params = sum([p.numel() for p in self.parameters()])

    def forward(self, x, fix_mask=None, **kwargs):
        x = x.transpose(2, 1)
        
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat

    def __str__(self):
        return f"{self.__class__.__name__} ({self.num_params} params)"


class ShallowPointNetCls(nn.Module):
    """
    Like `PointNetCls`, but less layers.
    Default has 500k parameters.
    :param k: number of classes.
    :param d: dimension of the points of the point cloud (e.g. 2 or 3).
    """

    def __init__(
        self,
        k=2,
        d=2,
        d_feat=512,
        d_hidden_1=512,
        stn_1=512,
        stn_2=256,
        feature_transform=False,
    ):
        super(ShallowPointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = ShallowPointNetfeat(
            global_feat=True,
            feature_transform=feature_transform,
            d=d,
            d_out=d_feat,
            stn_1=stn_1,
            stn_2=stn_2,
        )
        self.fc1 = nn.Linear(d_feat, d_hidden_1)
        self.fc2 = nn.Linear(d_hidden_1, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(d_hidden_1)
        self.relu = nn.ReLU()
        self.num_params = sum([p.numel() for p in self.parameters()])

    def forward(self, x):
        x = x.transpose(2, 1)
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1), trans, trans_feat

    def __str__(self):
        return f"{self.__class__.__name__} ({self.num_params} params)"


class PointNetDenseCls(nn.Module):
    """
    Network with input point cloud in dimension d and output point cloud in dimension k.
    Defaults to 1.7m parameters.
    """

    def __init__(
        self,
        k=2,
        d=2,
        feature_transform=False,
        d_feat=1024,
        stn_1=1024,
        stn_2=512,
        stn_3=256,
        d_conv1=512,
        d_conv2=256,
        d_conv3=128,
    ):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(
            global_feat=False,
            feature_transform=feature_transform,
            d=d,
            d_out=d_feat,
            stn_1=stn_1,
            stn_2=stn_2,
            stn_3=stn_3,
        )
        self.conv1 = torch.nn.Conv1d(64 + d_feat, d_conv1, 1)
        self.conv2 = torch.nn.Conv1d(d_conv1, d_conv2, 1)
        self.conv3 = torch.nn.Conv1d(d_conv2, d_conv3, 1)
        self.conv4 = torch.nn.Conv1d(d_conv3, self.k, 1)
        self.bn1 = nn.BatchNorm1d(d_conv1)
        self.bn2 = nn.BatchNorm1d(d_conv2)
        self.bn3 = nn.BatchNorm1d(d_conv3)
        self.num_params = sum([p.numel() for p in self.parameters()])

    def forward(self, x):
        x = x.transpose(2, 1)
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()
        x = F.log_softmax(x.view(-1, self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans, trans_feat

    def __str__(self):
        return f"{self.__class__.__name__} ({self.num_params} params)"


class ShallowPointNetDenseCls(nn.Module):
    """
    Like `PointNetDenseCls`, but less layers.
    Defaults to 400k parameters.
    """

    def __init__(
        self,
        k=2,
        d=2,
        feature_transform=False,
        d_feat=512,
        stn_1=512,
        stn_2=256,
        d_conv1=256,
        d_conv2=64,
    ):
        super(ShallowPointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform = feature_transform
        self.feat = ShallowPointNetfeat(
            global_feat=False,
            feature_transform=feature_transform,
            d=d,
            d_out=d_feat,
            stn_1=stn_1,
            stn_2=stn_2,
        )
        self.conv1 = torch.nn.Conv1d(64 + d_feat, d_conv1, 1)
        self.conv2 = torch.nn.Conv1d(d_conv1, d_conv2, 1)
        self.conv3 = torch.nn.Conv1d(d_conv2, self.k, 1)
        self.bn1 = nn.BatchNorm1d(d_conv1)
        self.bn2 = nn.BatchNorm1d(d_conv2)
        self.num_params = sum([p.numel() for p in self.parameters()])

    def forward(self, x):
        x = x.transpose(2, 1)
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        x = x.transpose(2, 1).contiguous()
        x = F.log_softmax(x.view(-1, self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans, trans_feat

    def __str__(self):
        return f"{self.__class__.__name__} ({self.num_params} params)"


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(
        torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2))
    )
    return loss


if __name__ == "__main__":
    # Examples from the original code base. Might not run.
    """
    sim_data = Variable(torch.rand(32,3,2500))
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = STNkd(k=64)
    out = trans(sim_data_64d)
    print('stn64d', out.size())
    print('loss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k = 5)
    out, _, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(k = 3)
    out, _, _ = seg(sim_data)
    print('seg', out.size())
    """
