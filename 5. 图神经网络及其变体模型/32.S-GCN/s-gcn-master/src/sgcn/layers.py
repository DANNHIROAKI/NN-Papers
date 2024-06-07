import torch
import torch.nn as nn


class SphericalConvolution(nn.Module):
    def __init__(self, in_features, out_features, order, non_linearity, bias=True):
        assert non_linearity in ['relu', 'elu', 'sigmoid', 'tanh', 'mish', 'none'], 'Incorrect non-linearity'

        super(SphericalConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.order_squared = order ** 2

        self.non_linearity = None
        self.non_linearity_name = non_linearity
        if non_linearity == 'relu':
            self.non_linearity = nn.functional.relu
        elif non_linearity == 'elu':
            self.non_linearity = nn.functional.elu
        elif non_linearity == 'sigmoid':
            self.non_linearity = nn.functional.sigmoid
        elif non_linearity == 'tahn':
            self.non_linearity = nn.functional.tanh
        elif non_linearity == 'none':
            self.non_linearity = None

        self.bias = nn.Parameter(torch.Tensor(out_features)) if bias else None
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weights = nn.ParameterList([
            nn.Parameter(torch.Tensor(in_features, out_features))
            for i in range(self.order_squared)
        ])

        torch.nn.init.xavier_normal_(self.weight)
        for i in range(self.order_squared):
            torch.nn.init.xavier_normal_(self.weights[i])
        if bias:
            torch.nn.init.constant_(self.bias, 0)

    def forward(self, X, sh):
        x_part = X.matmul(self.weight)
        conv_sh = []
        for i in range(self.order_squared):
            conv_sh.append(sh[i].mm(X).mm(self.weights[i]))
        res = x_part + torch.stack(conv_sh).sum(axis=0)

        if self.bias is not None:
            res = res + self.bias.expand(len(X), self.out_features)
        if self.non_linearity is not None:
            return self.non_linearity(res)
        else:
            return res
