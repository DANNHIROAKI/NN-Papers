import torch
import torch.nn as nn

from src.sgcn import layers

RESIDUE_TYPES_N = 20


def get_network(name, features_dim, order, conv_nonlinearity, dropout):
    return globals()[name](features_dim, order, conv_nonlinearity, dropout)


# S-GCN
class S5AndDropout(nn.Module):
    def __init__(self, features_dim, order=5, non_linearity='elu', dropout=0.2):
        super(S5AndDropout, self).__init__()
        self.conv_dropout = nn.Dropout(dropout)
        self.conv0 = layers.SphericalConvolution(RESIDUE_TYPES_N + features_dim, 20, order, non_linearity)
        self.conv1 = layers.SphericalConvolution(20, 16, order, non_linearity)
        self.batch_norm1 = nn.BatchNorm1d(16)
        self.conv2 = layers.SphericalConvolution(16, 8, order, non_linearity)
        self.conv3 = layers.SphericalConvolution(8, 4, order, non_linearity)
        self.batch_norm2 = nn.BatchNorm1d(4)
        self.conv4 = layers.SphericalConvolution(4, 1, order, 'none')

    def forward(self, one_hot, features, sh):
        x = torch.cat([one_hot, features], dim=1)
        x = self.conv_dropout(self.conv0(x, sh))
        x = self.conv_dropout(self.batch_norm1(self.conv1(x, sh)))
        x = self.conv_dropout(self.conv2(x, sh))
        x = self.conv_dropout(self.batch_norm2(self.conv3(x, sh)))
        x = self.conv4(x, sh)
        x = torch.sigmoid(x)
        return x


# S-GCN_s
class S5WithScoring(nn.Module):
    def __init__(self, features_dim, order=5, non_linearity='elu', dropout=0.2):
        super(S5WithScoring, self).__init__()
        self.conv_dropout = nn.Dropout(dropout)
        self.conv0 = layers.SphericalConvolution(RESIDUE_TYPES_N + features_dim, 20, order, non_linearity)
        self.conv1 = layers.SphericalConvolution(20, 16, order, non_linearity)
        self.batch_norm1 = nn.BatchNorm1d(16)
        self.conv2 = layers.SphericalConvolution(16, 14, order, non_linearity)
        self.conv3 = layers.SphericalConvolution(14, 12, order, non_linearity)
        self.batch_norm2 = nn.BatchNorm1d(12)
        self.conv4 = layers.SphericalConvolution(12, 8, order, non_linearity)

        self.fc1 = nn.Linear(8, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.elu = nn.ELU()
        self.fc_drp = nn.Dropout(0.1)

    def forward(self, one_hot, features, sh):
        x = torch.cat([one_hot, features], dim=1)
        x = self.conv_dropout(self.conv0(x, sh))
        x = self.conv_dropout(self.batch_norm1(self.conv1(x, sh)))
        x = self.conv_dropout(self.conv2(x, sh))
        x = self.conv_dropout(self.batch_norm2(self.conv3(x, sh)))
        x = self.conv4(x, sh)
        x = self.fc_drp(self.elu(self.fc1(x)))
        x = self.fc_drp(self.elu(self.fc2(x)))
        x = torch.sigmoid(self.fc3(x))
        return x
