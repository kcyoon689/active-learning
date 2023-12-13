'''Loss Prediction Module in PyTorch.

Reference:
[Yoo et al. 2019] Learning Loss for Active Learning (https://arxiv.org/abs/1905.03677)
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class LossNet(nn.Module):
    def __init__(self, feature_sizes=[19, 38, 1], num_channels=[1024, 512, 256], interm_dim=128):
        super(LossNet, self).__init__()

        self.GAP1 = nn.AvgPool2d(feature_sizes[0])
        self.GAP2 = nn.AvgPool2d(feature_sizes[1])
        self.GAP3 = nn.AvgPool2d(feature_sizes[2])

        self.FC1 = nn.Linear(num_channels[0], interm_dim)
        self.FC2 = nn.Linear(num_channels[1], interm_dim)
        self.FC3 = nn.Linear(num_channels[2], interm_dim)

        self.linear = nn.Linear(3 * interm_dim, 1)

    def forward(self, feats, feat4_3, feat_extra):
        out1 = self.GAP1(feats)
        out1 = out1.view(out1.size(0), -1)
        out1 = F.relu(self.FC1(out1))

        out2 = self.GAP2(feat4_3)
        out2 = out2.view(out2.size(0), -1)
        out2 = F.relu(self.FC2(out2))

        out3 = self.GAP3(feat_extra)
        out3 = out3.view(out3.size(0), -1)
        out3 = F.relu(self.FC3(out3))

        out = self.linear(torch.cat((out1, out2, out3), 1))
        return out
