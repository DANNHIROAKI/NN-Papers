import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.baseline import layers

ATOM_TYPES_N = 167
RESIDUE_TYPES_N = 20
GEMME_FEATURES_N = 20


def get_network_legacy(name, features_dim, n_channels_r, conv_nonlinearity):
    return globals()[name](features_dim, n_channels_r, conv_nonlinearity)


def get_network(name, features_dim, n_channels_r, enc_layers,
                      conv_layers, scoring_layers, conv_nonlinearity, dropout):
    return globals()[name](features_dim, n_channels_r, enc_layers, conv_layers, scoring_layers, conv_nonlinearity, dropout)


class L1(nn.Module):
    def __init__(self, features_dim, n_channels_r, non_linearity):
        super(L1, self).__init__()
        self.l_conv1 = layers.GraphConvSparseOne(RESIDUE_TYPES_N + features_dim, 16, n_channels_r, non_linearity)
        self.l_conv2 = layers.GraphConvSparseOne(16, 8, n_channels_r, non_linearity)
        self.l_conv3 = layers.GraphConvSparseOne(8, 4, n_channels_r, non_linearity)
        self.l_conv4 = layers.GraphConvSparseOne(4, 1, n_channels_r, non_linearity)

    def forward(self, one_hot, features, gemme_features, a_res):
        x = torch.cat([one_hot, features], dim=1)
        x = self.l_conv1(x, a_res)
        x = self.l_conv2(x, a_res)
        x = self.l_conv3(x, a_res)
        x = self.l_conv4(x, a_res)
        return torch.mean(x).squeeze()


class L2(nn.Module):
    def __init__(self, features_dim, n_channels_r, non_linearity):
        super(L2, self).__init__()
        self.l_conv1 = layers.GraphConvSparseOne(RESIDUE_TYPES_N + features_dim, 16, n_channels_r, non_linearity)
        self.l_conv2 = layers.GraphConvSparseOne(16, 8, n_channels_r, non_linearity)
        self.l_conv3 = layers.GraphConvSparseOne(8, 4, n_channels_r, non_linearity)
        self.l_conv4 = layers.GraphConvSparseOne(4, 1, n_channels_r, non_linearity)
        self.l_conv1d_1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=9)
        self.l_conv1d_2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5)
        self.l_conv1d_3 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3)

    def forward(self, one_hot, features, gemme_features, a_res):
        x = torch.cat([one_hot, features], dim=1)
        x = self.l_conv1(x, a_res)
        x = self.l_conv2(x, a_res)
        x = self.l_conv3(x, a_res)
        x = self.l_conv4(x, a_res)
        x = self.l_conv1d_1(x.squeeze().expand((1, 1, -1)))
        x = self.l_conv1d_2(x)
        x = self.l_conv1d_3(x).view(-1).squeeze()
        return torch.mean(x).squeeze()


class L3(nn.Module):
    def __init__(self, features_dim, n_channels_r, non_linearity):
        super(L3, self).__init__()
        print('DIM:', RESIDUE_TYPES_N + features_dim)
        self.l_conv1 = layers.GraphConvSparseOne(RESIDUE_TYPES_N + features_dim, 20, n_channels_r, non_linearity)
        self.dropout1 = nn.Dropout(0.3)
        self.l_conv2 = layers.GraphConvSparseOne(20, 16, n_channels_r, non_linearity)
        self.dropout2 = nn.Dropout(0.3)
        self.l_conv3 = layers.GraphConvSparseOne(16, 12, n_channels_r, non_linearity)
        self.dropout3 = nn.Dropout(0.3)
        self.l_conv4 = layers.GraphConvSparseOne(12, 8, n_channels_r, non_linearity)
        self.dropout4 = nn.Dropout(0.3)
        self.l_conv5 = layers.GraphConvSparseOne(8, 4, n_channels_r, non_linearity)
        self.dropout5 = nn.Dropout(0.3)
        self.linear1 = nn.Linear(4, 8)
        self.linear2 = nn.Linear(8, 4)
        self.linear3 = nn.Linear(4, 1)

    def forward(self, one_hot, features, gemme_features, a_res):
        x = torch.cat([one_hot, features], dim=1)
        x = self.l_conv1(x, a_res)
        x = self.dropout1(x)
        x = self.l_conv2(x, a_res)
        x = self.dropout2(x)
        x = self.l_conv3(x, a_res)
        x = self.dropout3(x)
        x = self.l_conv4(x, a_res)
        x = self.dropout4(x)
        x = self.l_conv5(x, a_res)
        x = self.dropout5(x)
        x = F.elu(self.linear1(x))
        x = F.elu(self.linear2(x))
        x = F.sigmoid(self.linear3(x))
        return x


class L4(nn.Module):
    def __init__(self, features_dim, n_channels_r, non_linearity):
        super(L4, self).__init__()
        print('DIM:', RESIDUE_TYPES_N + features_dim)
        self.dropout = nn.Dropout(0.1)
        self.l_conv1 = layers.GraphConvSparseOne(RESIDUE_TYPES_N + features_dim, 20, n_channels_r, non_linearity)
        self.l_conv2 = layers.GraphConvSparseOne(20, 16, n_channels_r, non_linearity)
        self.l_conv3 = layers.GraphConvSparseOne(16, 8, n_channels_r, non_linearity)
        self.linear1 = nn.Linear(8, 16)
        self.linear2 = nn.Linear(16, 8)
        self.linear3 = nn.Linear(8, 1)

    def forward(self, one_hot, features, gemme_features, a_res):
        x = torch.cat([one_hot, features], dim=1)
        x = self.l_conv1(x, a_res)
        x = self.dropout(x)
        x = self.l_conv2(x, a_res)
        x = self.dropout(x)
        x = self.l_conv3(x, a_res)
        x = self.dropout(x)
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.dropout(F.relu(self.linear2(x)))
        x = F.sigmoid(self.linear3(x))
        return x


class L5(nn.Module):
    def __init__(self, features_dim, n_channels_r, non_linearity):
        super(L5, self).__init__()
        print('L5 DIM:', RESIDUE_TYPES_N + features_dim)
        # Encoder
        self.enc_lin1 = nn.Linear(RESIDUE_TYPES_N + features_dim, 32, bias=True)
        # self.enc_bn1 = nn.BatchNorm1d(32)
        self.enc_drp1 = nn.Dropout(0.3)
        self.enc_lin2 = nn.Linear(32, 64, bias=True)
        # self.enc_bn2 = nn.BatchNorm1d(64)
        self.enc_drp2 = nn.Dropout(0.3)
        self.enc_lin3 = nn.Linear(64, 128, bias=True)
        # self.enc_bn3 = nn.BatchNorm1d(128)
        self.enc_drp3 = nn.Dropout(0.3)

        # Message passing
        self.g_conv1 = layers.GraphConvSparseOne(128, 64, n_channels_r, non_linearity)
        self.g_drp1 = nn.Dropout(0.1)
        self.g_conv2 = layers.GraphConvSparseOne(64, 32, n_channels_r, non_linearity)
        self.g_drp2 = nn.Dropout(0.1)
        self.g_conv3 = layers.GraphConvSparseOne(32, 16, n_channels_r, non_linearity)
        self.g_drp3 = nn.Dropout(0.1)
        self.g_conv4 = layers.GraphConvSparseOne(16, 8, n_channels_r, non_linearity)
        self.g_drp4 = nn.Dropout(0.1)
        self.g_conv5 = layers.GraphConvSparseOne(8, 4, n_channels_r, non_linearity)
        self.g_drp5 = nn.Dropout(0.1)

        # Final
        self.final_dropout = nn.Dropout(0.1)
        self.fin_lin1 = nn.Linear(4, 8)
        # self.fin_bn1 = nn.BatchNorm1d(8)
        self.fin_lin2 = nn.Linear(8, 4)
        # self.fin_bn2 = nn.BatchNorm1d(4)
        self.fin_lin3 = nn.Linear(4, 1)

    def forward(self, one_hot, features, gemme_features, a_res):
        x = torch.cat([one_hot, features], dim=1)
        x = self.enc_drp1(F.elu(self.enc_lin1(x)))
        x = self.enc_drp2(F.elu(self.enc_lin2(x)))
        x = self.enc_drp3(F.elu(self.enc_lin3(x)))

        x = self.g_drp1(self.g_conv1(x, a_res))
        x = self.g_drp2(self.g_conv2(x, a_res))
        x = self.g_drp3(self.g_conv3(x, a_res))
        x = self.g_drp4(self.g_conv4(x, a_res))
        x = self.g_drp5(self.g_conv5(x, a_res))

        x = self.final_dropout(F.elu(self.fin_lin1(x)))
        x = self.final_dropout(F.elu(self.fin_lin2(x)))
        x = torch.sigmoid(self.fin_lin3(x))

        return x


class L6(nn.Module):
    def __init__(self, features_dim, n_channels_r, non_linearity):
        super(L6, self).__init__()
        print('L6 DIM:', RESIDUE_TYPES_N + features_dim)
        # Encoder
        self.enc_lin1 = nn.Linear(RESIDUE_TYPES_N + features_dim, 32, bias=True)
        self.enc_drp1 = nn.Dropout(0.1)
        self.enc_lin2 = nn.Linear(32, 64, bias=True)
        self.enc_drp2 = nn.Dropout(0.3)
        self.enc_lin3 = nn.Linear(64, 128, bias=True)
        self.enc_drp3 = nn.Dropout(0.3)
        self.enc_lin4 = nn.Linear(128, 256, bias=True)
        self.enc_drp4 = nn.Dropout(0.3)

        # Message passing
        self.g_conv0 = layers.GraphConvSparseOne(256, 128, n_channels_r, non_linearity)
        self.g_drp0 = nn.Dropout(0.3)
        self.g_conv1 = layers.GraphConvSparseOne(128, 64, n_channels_r, non_linearity)
        self.g_drp1 = nn.Dropout(0.3)
        self.g_conv2 = layers.GraphConvSparseOne(64, 32, n_channels_r, non_linearity)
        self.g_drp2 = nn.Dropout(0.3)
        self.g_conv3 = layers.GraphConvSparseOne(32, 16, n_channels_r, non_linearity)
        self.g_drp3 = nn.Dropout(0.3)
        self.g_conv4 = layers.GraphConvSparseOne(16, 8, n_channels_r, non_linearity)
        self.g_drp4 = nn.Dropout(0.1)
        self.g_conv5 = layers.GraphConvSparseOne(8, 4, n_channels_r, non_linearity)

        # Final
        self.final_droput = nn.Dropout(0.3)
        self.fin_lin1 = nn.Linear(4, 8)
        # self.fin_bn1 = nn.BatchNorm1d(8)
        self.fin_lin2 = nn.Linear(8, 16)
        self.fin_lin3 = nn.Linear(16, 8)
        self.fin_lin4 = nn.Linear(8, 4)
        # self.fin_bn2 = nn.BatchNorm1d(4)
        self.fin_lin5 = nn.Linear(4, 1)

    def forward(self, one_hot, features, gemme_features, a_res):
        x = torch.cat([one_hot, features], dim=1)
        x = self.enc_drp1(F.elu(self.enc_lin1(x)))
        x = self.enc_drp2(F.elu(self.enc_lin2(x)))
        x = self.enc_drp3(F.elu(self.enc_lin3(x)))
        x = self.enc_drp4(F.elu(self.enc_lin4(x)))

        x = self.g_drp0(self.g_conv0(x, a_res))
        x = self.g_drp1(self.g_conv1(x, a_res))
        x = self.g_drp2(self.g_conv2(x, a_res))
        x = self.g_drp3(self.g_conv3(x, a_res))
        x = self.g_drp4(self.g_conv4(x, a_res))
        x = self.g_conv5(x, a_res)

        x = self.final_droput(F.elu(self.fin_lin1(x)))
        x = self.final_droput(F.elu(self.fin_lin2(x)))
        x = self.final_droput(F.elu(self.fin_lin3(x)))
        x = self.final_droput(F.elu(self.fin_lin4(x)))
        x = torch.sigmoid(self.fin_lin5(x))

        return x


class L7(nn.Module):
    def __init__(self, features_dim, n_channels_r, non_linearity, dropout_level=0.3):
        super(L7, self).__init__()
        print('L7 DIM:', RESIDUE_TYPES_N + features_dim)
        self.droput = nn.Dropout(dropout_level)
        if non_linearity == 'elu':
            self.activation = F.elu
        elif non_linearity == 'relu':
            self.activation = F.relu
        # Encoder
        self.enc_lin1 = nn.Linear(RESIDUE_TYPES_N + features_dim, 32, bias=True)
        self.enc_lin2 = nn.Linear(32, 64, bias=True)
        self.enc_lin3 = nn.Linear(64, 128, bias=True)
        self.enc_lin4 = nn.Linear(128, 256, bias=True)

        # Message passing
        self.g_conv0 = layers.GraphConvSparseOne(256, 200, n_channels_r, non_linearity)
        self.g_conv1 = layers.GraphConvSparseOne(200, 150, n_channels_r, non_linearity)
        self.g_conv2 = layers.GraphConvSparseOne(150, 128, n_channels_r, non_linearity)
        self.g_conv3 = layers.GraphConvSparseOne(128, 100, n_channels_r, non_linearity)
        self.g_conv4 = layers.GraphConvSparseOne(100, 64, n_channels_r, non_linearity)
        self.g_conv5 = layers.GraphConvSparseOne(64, 32, n_channels_r, non_linearity)
        self.g_conv6 = layers.GraphConvSparseOne(32, 16, n_channels_r, non_linearity)
        self.g_conv7 = layers.GraphConvSparseOne(16, 8, n_channels_r, non_linearity)

        # Final
        self.fin_lin1 = nn.Linear(8, 16)
        self.fin_lin2 = nn.Linear(16, 32)
        self.fin_lin3 = nn.Linear(32, 16)
        self.fin_lin4 = nn.Linear(16, 8)
        self.fin_lin5 = nn.Linear(8, 1)

    def forward(self, one_hot, features, gemme_features, a_res):
        x = torch.cat([one_hot, features], dim=1)
        x = self.droput(self.activation(self.enc_lin1(x)))
        x = self.droput(self.activation(self.enc_lin2(x)))
        x = self.droput(self.activation(self.enc_lin3(x)))
        x = self.droput(self.activation(self.enc_lin4(x)))

        x = self.droput(self.g_conv0(x, a_res))
        x = self.droput(self.g_conv1(x, a_res))
        x = self.droput(self.g_conv2(x, a_res))
        x = self.droput(self.g_conv3(x, a_res))
        x = self.droput(self.g_conv4(x, a_res))
        x = self.droput(self.g_conv5(x, a_res))
        x = self.droput(self.g_conv6(x, a_res))
        x = self.droput(self.g_conv7(x, a_res))

        x = self.droput(self.activation(self.fin_lin1(x)))
        x = self.droput(self.activation(self.fin_lin2(x)))
        x = self.droput(self.activation(self.fin_lin3(x)))
        x = self.droput(self.activation(self.fin_lin4(x)))
        x = torch.sigmoid(self.fin_lin5(x))
        return x


class L8(nn.Module):
    def __init__(self, features_dim, n_channels_r, non_linearity, dropout_level=0.3):
        super(L8, self).__init__()
        print('L8 DIM:', RESIDUE_TYPES_N + features_dim)
        self.droput = nn.Dropout(dropout_level)
        if non_linearity == 'elu':
            self.activation = F.elu
        elif non_linearity == 'relu':
            self.activation = F.relu
        # Encoder
        self.enc_lin1 = nn.Linear(RESIDUE_TYPES_N + features_dim, 32, bias=True)
        self.enc_lin2 = nn.Linear(32, 64, bias=True)
        self.enc_lin3 = nn.Linear(64, 128, bias=True)
        self.enc_lin4 = nn.Linear(128, 256, bias=True)

        # Message passing
        self.g_conv0 = layers.GraphConvSparseOne(256, 200, n_channels_r, non_linearity)
        self.g_conv1 = layers.GraphConvSparseOne(200, 150, n_channels_r, non_linearity)
        self.g_conv2 = layers.GraphConvSparseOne(150, 128, n_channels_r, non_linearity)
        self.g_conv3 = layers.GraphConvSparseOne(128, 100, n_channels_r, non_linearity)
        self.g_conv4 = layers.GraphConvSparseOne(100, 64, n_channels_r, non_linearity)
        self.g_conv5 = layers.GraphConvSparseOne(64, 48, n_channels_r, non_linearity)
        self.g_conv6 = layers.GraphConvSparseOne(48, 32, n_channels_r, non_linearity)
        self.g_conv7 = layers.GraphConvSparseOne(32, 16, n_channels_r, non_linearity)
        self.g_conv8 = layers.GraphConvSparseOne(16, 8, n_channels_r, non_linearity)

        # Final
        self.fin_lin1 = nn.Linear(8, 16)
        self.fin_lin2 = nn.Linear(16, 32)
        self.fin_lin3 = nn.Linear(32, 16)
        self.fin_lin4 = nn.Linear(16, 8)
        self.fin_lin5 = nn.Linear(8, 1)

    def forward(self, one_hot, features, gemme_features, a_res):
        x = torch.cat([one_hot, features], dim=1)
        x = self.droput(self.activation(self.enc_lin1(x)))
        x = self.droput(self.activation(self.enc_lin2(x)))
        x = self.droput(self.activation(self.enc_lin3(x)))
        x = self.droput(self.activation(self.enc_lin4(x)))

        x = self.droput(self.g_conv0(x, a_res))
        x = self.droput(self.g_conv1(x, a_res))
        x = self.droput(self.g_conv2(x, a_res))
        x = self.droput(self.g_conv3(x, a_res))
        x = self.droput(self.g_conv4(x, a_res))
        x = self.droput(self.g_conv5(x, a_res))
        x = self.droput(self.g_conv6(x, a_res))
        x = self.droput(self.g_conv7(x, a_res))
        x = self.droput(self.g_conv8(x, a_res))

        x = self.droput(self.activation(self.fin_lin1(x)))
        x = self.droput(self.activation(self.fin_lin2(x)))
        x = self.droput(self.activation(self.fin_lin3(x)))
        x = self.droput(self.activation(self.fin_lin4(x)))
        x = torch.sigmoid(self.fin_lin5(x))
        return x


class LightNetworkEnc(nn.Module):
    def __init__(self, features_dim, n_channels_r, enc_layers, conv_layers, scoring_layers, non_linearity, dropout_level=0.3):
        super(LightNetworkEnc, self).__init__()
        # print(f'LightNetworkEnc Encoder: {enc_layers}\tConv: {conv_layers}\tScoring: {scoring_layers}\tDim: {RESIDUE_TYPES_N + features_dim}')
        self.dropout = nn.Dropout(dropout_level)
        if non_linearity == 'elu':
            self.activation = F.elu
        elif non_linearity == 'relu':
            self.activation = F.relu

        self.encoders = []
        dim = RESIDUE_TYPES_N + features_dim
        for i in range(enc_layers):
            self.encoders.append(nn.Linear(dim, 2 ** (i + 5), bias=True))
            dim = 2 ** (i + 5)
        self.encoders = nn.ModuleList(self.encoders)

        step = (dim - 8) // conv_layers
        self.conv = []
        for i in range(conv_layers - 1):
            self.conv.append(layers.GraphConvSparseOne(dim, dim - step, n_channels_r, non_linearity))
            dim -= step
        self.conv.append(layers.GraphConvSparseOne(dim, 8, n_channels_r, non_linearity))
        self.conv = nn.ModuleList(self.conv)

        dim = 8
        self.scoring = []
        for i in range(scoring_layers):
            self.scoring.append(nn.Linear(dim, dim * 2, bias=True))
            dim *= 2
        for i in range(scoring_layers - 1):
            self.scoring.append(nn.Linear(dim, dim // 2, bias=True))
            dim //= 2
        self.scoring = nn.ModuleList(self.scoring)

        self.final = nn.Linear(dim, 1)

    def forward(self, one_hot, features, gemme_features, a_res):
        x = torch.cat([one_hot, features], dim=1)
        for layer in self.encoders:
            x = self.dropout(self.activation(layer(x)))

        for layer in self.conv:
            x = self.dropout(layer(x, a_res))

        for layer in self.scoring:
            x = self.dropout(self.activation(layer(x)))

        x = torch.sigmoid(self.final(x))

        return x
