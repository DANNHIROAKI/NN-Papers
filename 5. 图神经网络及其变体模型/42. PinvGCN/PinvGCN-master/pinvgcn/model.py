
import torch
import torch.nn.functional as F
import torch_geometric
import numpy as np



def get_coefficient_preset(name, alpha=1, beta=1, gamma=1):
    r"""
    Get a list of (alpha,beta,gamma) tuples that describe the spectral filter
    basis functions given by a preset name. Each basis function may consist
    of a zero-impulse part, a pseudoinverse part, and a high-pass part. Each 
    part may be scaled manually by the alpha,beta,gamma arguments.
    The preset names are as follows:
    * 'single': one basis function combining all three parts
    * 'independent-parts': three independent basis functions, one for each part
    * 'no-zero-impulse': two basis functions, one for pseudoinverse and one for
    high-pass
    * 'no-high-pass': two basis functions, one for zero-impulse and one for
    pseudoinverse
    * 'independent-zero-impulse', two basis functions, one for zero-impulse and 
    one for combined pseudoinverse and high-pass
    * 'independent-pseudoinverse', two basis functions, one for pseudoinverse
    and one for combined zero-impulse and high-pass
    * 'independent-high-pass', two basis functions, one for high-pass and one
    for combined zero-impulse and pseudoinverse
    """
    if name == 'single':
        return [(alpha,beta,gamma)]
    elif name == 'independent-parts':
        return [(alpha,0,0), (0,beta,0),(0,0,gamma)]
    elif name == 'only-pseudoinverse':
        return [(0,beta,0)]
    elif name == 'no-zero-impulse':
        return [(0,beta,0), (0,0,gamma)]
    elif name == 'no-high-pass':
        return [(alpha,0,0), (0,beta,0)]
    elif name == 'independent-zero-impulse':
        return [(alpha,0,0), (0,beta,gamma)]
    elif name == 'independent-pseudoinverse':
        return [(alpha,0,gamma), (0,beta,0)]
    elif name == 'independent-high-pass':
        return [(alpha,beta,0), (0,0,gamma)]
    else:
        raise ValueError("Unknown coefficient preset: {}".format(name))



def get_filter_tensors(coeffs, data):
    result = []
    for alpha, beta, gamma in coeffs:
        s = 0
        if alpha != 0 or gamma != 0:
            s += ((alpha - data.eigengap*gamma) * data.zero_U) @ data.zero_U.T
        if beta != 0 or gamma != 0:
            s += (data.eigengap*(beta - gamma/data.nonzero_w) * data.nonzero_U) @ data.nonzero_U.T
        if gamma != 0:
            s += data.eigengap * gamma * torch.eye(data.num_nodes, data.num_nodes, dtype=torch.float, device=data.nonzero_w.device)
        result.append(s)
    return result


def print_filter_values(coeffs, data):
    for i, c in enumerate(coeffs):
        alpha, beta, gamma = c
        print("Filter basis function #{}:".format(i+1))
        print(" - zero-impulse part: {:.3f}".format(alpha))
        with np.printoptions(precision=3, suppress=True):
            print(" - pseudoinverse part: ", data.eigengap * beta / data.nonzero_w.cpu().numpy())
        print(" - high-pass part: {:.3f}".format(gamma * data.eigengap))



class ConvBase(torch.nn.Module):
    r"""
    Base class for layer modules that perform spectral convolution with a 
    possibly multi-dimensional filter space. This class provides the weight
    and bias parameters.
    """
    
    def __init__(self, in_channels, out_channels, num_weights, bias=True):
        super().__init__()

        self.num_weights = num_weights
        self.weights = []
        for k in range(num_weights):
            W = torch.nn.Parameter(torch.Tensor(in_channels, out_channels))
            self.weights.append(W)
            self.register_parameter("W{}".format(k), W)

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)


class PreconvolvedLinear(ConvBase):
    r"""
    Layer module that performs the remaining parts of convolutional layer when
    the original convolutions of the input with each basis function have
    already been precomputed. The forward operation expects the data object
    and a list of preconvolved input tensors.
    """
    
    def forward(self, data, preconvolved_input):
        Y = sum(preconvolved_input[i] @ self.weights[i] for i in range(self.num_weights))
        if self.bias is not None:
            Y += self.bias
        return Y

class PinvConv(ConvBase):
    r"""
    Layer module that performs convolution of its input with spectral filters 
    that may consist of a zero-impulse, a pseudoinverse, and a high-pass part.
    Its basis function coefficients are given in the argument `coeffs` as a 
    list of (alpha,beta,gamma) tuples, cf. `get_coefficient_preset`.
    If the `is_last` argument is True, the forward operation will only return 
    the output rows corresponding to training samples if the operation is in
    training mode.
    """
    
    def __init__(self, in_channels, out_channels, coeffs, bias=True, is_last=False):
        super().__init__(in_channels, out_channels, len(coeffs), bias)
        
        self.is_last = is_last
        self.coeffs = coeffs
        
    def forward(self, data, X):
        mask = data.train_mask if self.is_last and self.training else slice(None)
        
        zero_part = 0
        nonzero_part = 0
        result = 0
        
        for i in range(self.num_weights):
            Y = X @ self.weights[i]
            alpha, beta, gamma = self.coeffs[i]
            if alpha != 0 or gamma != 0:
                zero_part += (alpha - data.eigengap * gamma) * (data.zero_U.T @ Y)
            if beta != 0 or gamma != 0:
                nonzero_part += (beta / data.nonzero_w[:,np.newaxis] - gamma) * data.eigengap * (data.nonzero_U.T @ Y)
            if gamma != 0:
                result += data.eigengap * gamma * Y[mask, :]
        
        if torch.is_tensor(zero_part):
            result += data.zero_U[mask, :] @ zero_part
        if torch.is_tensor(nonzero_part):
            result += data.nonzero_U[mask, :] @ nonzero_part
        
        if self.bias is not None:
            result += self.bias
        return result



class PinvGCN(torch.nn.Module):
    r"""
    Full neural network with the Pseudoinverse GCN architecture, where each 
    layer performs convolution of its input with spectral filters that may 
    consist of a zero-impulse, a pseudoinverse, and a high-pass part.
    The basis function coefficients are given in the argument `coeffs` as a 
    list of (alpha,beta,gamma) tuples, cf. `get_coefficient_preset`.
    
    The convolution relies on the spectral data setup provided by 
    `setup_spectral_data`. For efficiency, the forward operation expects the
    precomputed convolutions of the original input, which can be set up with
    the `preconvolve_input` method.
    
    In training mode, the forward operation will only return a subset of 
    rows of the output corresponding to the training samples (because the 
    other rows are not required for loss computation).
    
    The `reset_parameters` method re-initializes the parameters of all layers.
    By default, Glorot initialization is used for the weight matrices and 
    zeros initialization for the biases. If required, you can control this 
    behaviour by the `init` argument to the constructor, which expects a list 
    holding either `'glorot'` or `'zeros'` for each basis function.
    """
    
    def __init__(self, coeffs, in_channels, out_channels, hidden_channels=[32], dropout=0.5, bias=True, init=None):
        
        super().__init__()
        
        self.depth = len(hidden_channels)+1
        self.dropout = None if dropout is None or dropout == 0 else torch.nn.Dropout(dropout)
        self.activation = torch.nn.ReLU(inplace=True)
        
        if isinstance(coeffs, str):
            coeffs = get_coefficient_preset(coeffs)
        self.coeffs = coeffs
        
        if init is None:
            init = len(coeffs) * ['glorot']
        elif isinstance(init, str):
            init = len(coeffs) * [init]
        self.init = init
        
        self.layers = torch.nn.ModuleList()
        self.layers.append(PreconvolvedLinear(in_channels, hidden_channels[0], len(coeffs), bias=bias))
        for i in range(self.depth-2):
            self.layers.append(PinvConv(hidden_channels[i], hidden_channels[i+1], coeffs, bias=bias))
        self.layers.append(PinvConv(hidden_channels[-1], out_channels, coeffs, bias=bias, is_last=True))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        r"""
        Re-initialize the parameters of all layers. Biases are always reset to
        zero. For weight matrices, either Glorot or zeros initialization is 
        used, depending on the value in the `init` argument of the constructor
        for the corresponding basis function.
        """
        for layer in self.layers:
            for i, W in enumerate(layer.weights):
                if self.init[i] == 'glorot':
                    torch_geometric.nn.inits.glorot(W)
                elif self.init[i] == 'zeros':
                    torch_geometric.nn.inits.zeros(W)
                else:
                    raise ValueError("Unknown initializer: {}".format(self.init[i]))
            torch_geometric.nn.inits.zeros(layer.bias)
    
    def split_parameters(self, weight_decay=5e-4):
        r"""
        Return a list of dictionaries, which can be given to an optimizer such
        that all weight matrices are equipped with the given weight decay
        while no weight decay is used for the biases.
        """
        return [
            dict(params=[W for l in self.layers for W in l.weights], weight_decay=weight_decay),
            dict(params=[l.bias for l in self.layers], weight_decay=0)
        ]
    
    def preconvolve_input(self, data, X):
        r"""
        Precompute the convolutions of the given input matrix with the model's
        basis function. The result can be passed to the network's forward 
        operation.
        """
        
        if X is None:
            return get_filter_tensors(self.coeffs, data)
        
        zero_U_X = data.zero_U.T @ X
        nonzero_U_X = data.nonzero_U.T @ X
        
        result = []
        for alpha, beta, gamma in self.coeffs:
            s = 0
            if alpha != 0 or gamma != 0:
                s += (alpha - gamma*data.eigengap) * (data.zero_U @ zero_U_X)
            if beta != 0 or gamma != 0:
                s += data.nonzero_U @ (data.eigengap * (beta/data.nonzero_w[:,np.newaxis] - gamma) * nonzero_U_X)
            if gamma != 0:
                s += data.eigengap * gamma * X
            result.append(s)
        return result
        
        # return [(0 if np.isclose(alpha, gamma*data.eigengap) else (alpha-gamma*data.eigengap)*data.zero_U @ zero_U_X) +
        #         (0 if beta == 0 and gamma == 0 else (beta/data.nonzero_w[:,np.newaxis] - gamma) * data.eigengap * data.nonzero_U @ nonzero_U_X) +
        #         (0 if gamma == 0 else data.eigengap * gamma * X)
        #     for alpha, beta, gamma in self.coeffs]
    
    
    def forward(self, data, preconvolved_input):
        X = self.layers[0](data, preconvolved_input)
        
        for l in self.layers[1:]:
            X = self.activation(X)
            if self.dropout is not None:
                X = self.dropout(X)
            X = l(data, X)
            
        return X
    
    def run_training(self, data, input, optimizer, num_epochs):
        self.train()
        y = data.y[data.train_mask]
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            out = self(data, input)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()
        return loss.item()
    
    def predict(self, data, input, mask=slice(None)):
        return self(data, input)[mask].max(dim=1)[1]
        
    def eval_accuracy(self, data, input):
        r"""Evaluate the accuracy of the trained network on the test set."""
        self.eval()
        correctness = self.predict(data, input, data.test_mask).eq(data.y[data.test_mask])
        # _, pred = self(data, input)[data.test_mask].max(dim=1)
        return float(correctness.sum().item()) / data.test_mask.sum().item()


    def average_absolute_weight_entries(self):
        return np.array([[np.abs(W.detach().cpu().numpy()).mean() for W in l.weights] for l in self.layers])