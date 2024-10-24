import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from models.pointnet_equi.layers import *


class STNkd(nn.Module):
    def __init__(self, args, d=64):
        super(STNkd, self).__init__()
        self.args = args
        
        self.conv1 = VNLinearLeakyReLU(d, 64//3, dim=4, negative_slope=0.0)
        self.conv2 = VNLinearLeakyReLU(64//3, 128//3, dim=4, negative_slope=0.0)
        self.conv3 = VNLinearLeakyReLU(128//3, 1024//3, dim=4, negative_slope=0.0)

        self.fc1 = VNLinearLeakyReLU(1024//3, 512//3, dim=3, negative_slope=0.0)
        self.fc2 = VNLinearLeakyReLU(512//3, 256//3, dim=3, negative_slope=0.0)
        
        if args.pooling == 'max':
            self.pool = VNMaxPool(1024//3)
        elif args.pooling == 'mean':
            self.pool = mean_pool
        
        self.fc3 = VNLinear(256//3, d)
        self.d = d

    def forward(self, x,equiv,proj):
        batchsize = x.size()[0]
        x = self.conv1(x,equiv,proj)
        x = self.conv2(x,equiv,proj)
        x = self.conv3(x,equiv,proj)
        x = self.pool(x)

        x = self.fc1(x,equiv,proj)
        x = self.fc2(x,equiv,proj)
        x = self.fc3(x,equiv,proj)
        
        return x

class STNkd_Dual(nn.Module):
    def __init__(self, args, d=64):
        super(STNkd_Dual, self).__init__()
        self.args = args
        
        self.conv1 = VNLinearLeakyReLU_Dual(d, 64//3, dim=4, negative_slope=0.0)
        self.conv2 = VNLinearLeakyReLU_Dual(64//3, 128//3, dim=4, negative_slope=0.0)
        self.conv3 = VNLinearLeakyReLU_Dual(128//3, 1024//3, dim=4, negative_slope=0.0)

        self.fc1 = VNLinearLeakyReLU_Dual(1024//3, 512//3, dim=3, negative_slope=0.0)
        self.fc2 = VNLinearLeakyReLU_Dual(512//3, 256//3, dim=3, negative_slope=0.0)
        
        if args.pooling == 'max':
            self.pool = VNMaxPool(1024//3)
        elif args.pooling == 'mean':
            self.pool = mean_pool
        
        self.fc3 = VNLinear_Dual(256//3, d)
        self.d = d

    def forward(self, x,equiv,proj):
        batchsize = x.size()[0]
        x,n1_ld,n1 = self.conv1(x,equiv,proj)
        
        x,n_ld,n = self.conv2(x,equiv,proj)
        n1_ld+=n_ld
        n1+=n
        x,n_ld,n = self.conv3(x,equiv,proj)
        n1_ld+=n_ld
        n1+=n
        x = self.pool(x)

        x,n_ld,n  = self.fc1(x,equiv,proj)
        n1_ld+=n_ld
        n1+=n
        x,n_ld,n  = self.fc2(x,equiv,proj)
        n1_ld+=n_ld
        n1+=n
        x,n_ld,n  = self.fc3(x,equiv,proj)
        n1_ld+=n_ld
        n1+=n
        return x,n1_ld,n1


class PointNetEncoder_Dual(nn.Module):
    def __init__(self, args, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder_Dual, self).__init__()
        self.args = args
        self.n_knn = 20
        
        self.conv_pos = VNLinearLeakyReLU_Dual(3, 64//3, dim=5, negative_slope=0.0)
        self.conv1 = VNLinearLeakyReLU_Dual(64//3, 64//3, dim=4, negative_slope=0.0)
        self.conv2 = VNLinearLeakyReLU_Dual(64//3*2, 128//3, dim=4, negative_slope=0.0)
        
        self.conv3 = VNLinear_Dual(128//3, 1024//3)
        self.bn3 = VNBatchNorm(1024//3, dim=4)
        
        self.std_feature = VNStdFeature_Dual(1024//3*2, dim=4, normalize_frame=False, negative_slope=0.0)
        
        if args.pooling == 'max':
            self.pool = VNMaxPool(64//3)
        elif args.pooling == 'mean':
            self.pool = mean_pool
        
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        
        if self.feature_transform:
            self.fstn = STNkd_Dual(args, d=64//3)
    '''
    def forward(self,x,equiv,proj):
        B, D, N = x.size()
        n_ld_list=[]
        n_list=[]

        x = x.unsqueeze(1)
        feat = get_graph_feature_cross(x, k=self.n_knn)
        x,n1_ld,n1 = self.conv_pos(feat,equiv,proj)
        x = self.pool(x)
        n_ld_list.append(n1_ld)
        n_list.append(n1)
        x,n_ld,n = self.conv1(x,equiv,proj)
        n_ld_list.append(n_ld)
        n_list.append(n)
        if self.feature_transform:
            x_global,n_ld,n = self.fstn(x,equiv,proj)#.unsqueeze(-1).repeat(1,1,1,N)
            x_global=x_global.unsqueeze(-1).repeat(1,1,1,N)
            n_ld_list.append(n_ld)
            n_list.append(n)
            x = torch.cat((x, x_global), 1)
        
        pointfeat = x
        x,n_ld,n = self.conv2(x,equiv,proj)
        n_ld_list.append(n_ld)
        n_list.append(n)
        x,n_ld,n=self.conv3(x,equiv,proj)
        x = self.bn3(x)
        n_ld_list.append(n_ld)
        n_list.append(n)

        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x, trans,n_ld,n = self.std_feature(x,equiv,proj)
        x = x.view(B, -1, N)
        n_ld_list.append(n_ld)
        n_list.append(n)
        
        x = torch.max(x, -1, keepdim=False)[0]
        
        trans_feat = None
        if self.global_feat:
            return x, trans, trans_feat,n_ld_list,n_list
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat,n_ld_list,n_list
    '''
    def forward(self, x,equiv,proj):
        B, D, N = x.size()
        
        x = x.unsqueeze(1)
        feat = get_graph_feature_cross(x, k=self.n_knn)
        x,n1_ld,n1 = self.conv_pos(feat,equiv,proj)
        x = self.pool(x)
        
        x,n_ld,n = self.conv1(x,equiv,proj)
        n1_ld+=n_ld
        n1+=n
        if self.feature_transform:
            x_global,n_ld,n = self.fstn(x,equiv,proj)#.unsqueeze(-1).repeat(1,1,1,N)
            x_global=x_global.unsqueeze(-1).repeat(1,1,1,N)
            n1_ld+=n_ld
            n1+=n
            x = torch.cat((x, x_global), 1)
        
        pointfeat = x
        x,n_ld,n = self.conv2(x,equiv,proj)
        n1_ld+=n_ld
        n1+=n
        x,n_ld,n=self.conv3(x,equiv,proj)
        x = self.bn3(x)
        n1_ld+=n_ld
        n1+=n

        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x, trans,n_ld,n = self.std_feature(x,equiv,proj)
        x = x.view(B, -1, N)
        n1_ld+=n_ld
        n1+=n
        
        x = torch.max(x, -1, keepdim=False)[0]
        
        trans_feat = None
        if self.global_feat:
            return x, trans, trans_feat,n1_ld,n1
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat,n1_ld,n1
    

class PointNetEncoder(nn.Module):
    def __init__(self, args, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.args = args
        self.n_knn = 20
        
        self.conv_pos = VNLinearLeakyReLU(3, 64//3, dim=5, negative_slope=0.0)
        self.conv1 = VNLinearLeakyReLU(64//3, 64//3, dim=4, negative_slope=0.0)
        self.conv2 = VNLinearLeakyReLU(64//3*2, 128//3, dim=4, negative_slope=0.0)
        
        self.conv3 = VNLinear(128//3, 1024//3)
        self.bn3 = VNBatchNorm(1024//3, dim=4)
        
        self.std_feature = VNStdFeature(1024//3*2, dim=4, normalize_frame=False, negative_slope=0.0)
        
        if args.pooling == 'max':
            self.pool = VNMaxPool(64//3)
        elif args.pooling == 'mean':
            self.pool = mean_pool
        
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        
        if self.feature_transform:
            self.fstn = STNkd(args, d=64//3)

    def forward(self, x,equiv,proj):
        B, D, N = x.size()
        
        x = x.unsqueeze(1)
        feat = get_graph_feature_cross(x, k=self.n_knn)
        x = self.conv_pos(feat,equiv,proj)
        x = self.pool(x)
        
        x = self.conv1(x,equiv,proj)
        
        if self.feature_transform:
            x_global = self.fstn(x,equiv,proj).unsqueeze(-1).repeat(1,1,1,N)
            x = torch.cat((x, x_global), 1)
        
        pointfeat = x
        x = self.conv2(x,equiv,proj)
        x = self.bn3(self.conv3(x,equiv,proj))
        
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x, trans = self.std_feature(x,equiv,proj)
        x = x.view(B, -1, N)
        
        x = torch.max(x, -1, keepdim=False)[0]
        
        trans_feat = None
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat
