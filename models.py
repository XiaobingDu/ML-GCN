# -*-coding:utf-8-*-

import torchvision.models as models
from torch.nn import Parameter
from util import *
import torch
import torch.nn as nn

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    #显示属性
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCNResnet(nn.Module):
    def __init__(self, model, num_classes, in_channel=300, t=0, adj_file=None):
        super(GCNResnet, self).__init__()

        #定义 Resnet-101
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        ) #1
        self.num_classes = num_classes
        self.pooling = nn.MaxPool2d(14, 14) #2

        #定义 GCN 2-layers
        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, 2048)
        self.relu = nn.LeakyReLU(0.2)

        _adj = gen_A(num_classes, t, adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float())
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, feature, inp):
        #以下三行代码完成Resnet特征提取
        #eq.3
        feature = self.features(feature) #1
        feature = self.pooling(feature) #2
        feature = feature.view(feature.size(0), -1) #view：feature shape = 2D

        #GCN：learning inter-dependent object classification
        inp = inp[0]
        # tensor.detach(): 从self.A中分离出来的adj, 此时的adj与A共享存储空间
        #adj与self.A的区别：adj没有梯度，self.A有梯度；在adj没有改变的情况下self.A可以反向求导，adj不可以
        #adj是首先计算好的，在training过程中不会改变
        adj = gen_adj(self.A).detach()
        x = self.gc1(inp, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)
        print('output shape....', x.shape)

        x = x.transpose(0, 1)
        x = torch.matmul(feature, x) #eq.4
        print('feature shape....', feature.shape)
        print('x shape....', x.shape)

        return x

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.gc1.parameters(), 'lr': lr},
                {'params': self.gc2.parameters(), 'lr': lr},
                ]



def gcn_resnet101(num_classes, t, pretrained=False, adj_file=None, in_channel=300):
    model = models.resnet101(pretrained=pretrained)
    return GCNResnet(model, num_classes, t=t, adj_file=adj_file, in_channel=in_channel)
