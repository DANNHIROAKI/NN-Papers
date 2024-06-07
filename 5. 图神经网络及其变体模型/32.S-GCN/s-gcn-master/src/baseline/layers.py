import torch
import torch.nn as nn
import torch.nn.functional as F

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))


class GraphConvSparseOne(nn.Module):
    def __init__(self, in_features, out_features, n_channels, non_linearity, bias=True):
        assert non_linearity in ['relu', 'elu', 'sigmoid', 'tahn'], 'Incorrect non-linearity'

        super(GraphConvSparseOne, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.adj_weights = nn.ParameterList([
            nn.Parameter(torch.Tensor(in_features, out_features))
            for i in range(n_channels)
        ])
        self.bias = nn.Parameter(torch.Tensor(out_features)) if bias else None

        self.n_channels = n_channels

        self.non_linearity = None
        self.non_linearity_name = non_linearity
        if non_linearity == 'relu':
            self.non_linearity = nn.functional.relu
        elif non_linearity == 'elu':
            self.non_linearity = nn.functional.elu
        elif non_linearity == 'sigmoid':
            self.non_linearity = nn.functional.sigmoid
        elif non_linearity == 'tahn':
            self.non_linearity = nn.functional.tahn

        torch.nn.init.xavier_normal_(self.weight)
        for i in range(self.n_channels):
            torch.nn.init.xavier_normal_(self.adj_weights[i])
        if bias:
            torch.nn.init.constant_(self.bias, 0)

    def forward(self, X, adjs):
        x_part = X.matmul(self.weight)
        adj_parts = [
            self.non_linearity(adj.mm(X).mm(adj_w))
            for adj, adj_w in zip(adjs, self.adj_weights)
        ]
        adj_part = torch.stack(adj_parts).sum(axis=0)
        total = x_part + adj_part
        if self.bias is not None:
            total = total + self.bias.expand(len(X), self.out_features)
        return self.non_linearity(total)
