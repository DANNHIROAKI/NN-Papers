
import numpy as np
from scipy.io import loadmat
import os

import torch
from torch_geometric.data import Data

from .data import setup_spectral_data


def load_from_matlab(dataset, filename, max_num_ev=None, sort=True):
    r"""
    Load x and y data and Laplacian eigeninformation computed in MATLAB.
    The file must be located in data/{dataset}/{filename}. If `max_num_ev` is
    not None, only that many eigenvalues from the stored decomposition are 
    used. Returns a tuple of eigenvalues, eigenvectors, features, and labels.
    """
    matfile = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', dataset, filename)
    try:
        matdata = loadmat(matfile)
        w = matdata['w'].flatten()
        U = matdata['U']
        x = matdata['x']
        y = matdata['y'].flatten()-1
    except:
        raise ValueError('Error reading MATLAB file ' + matfile) from None
    
    if max_num_ev is not None and w.size > max_num_ev:
        ind = np.argpartition(w, max_num_ev)[:max_num_ev]
        w = w[ind]
        U = U[:, ind]
        
    if sort:
        ind = np.argsort(w)
        w = w[ind]
        U = U[:,ind]
    
    return w, U, x, y


def setup_spectral_data_from_matlab(dataset, filename, max_num_ev=None, threshold=1e-3):
    r"""
    Setup a Data object based on x and y data and Laplacian eigeninformation 
    computed in MATLAB. The file must be located in data/{dataset}/{filename}. 
    If `max_num_ev` is not None, only that many eigenvalues from the stored 
    decomposition are used. Returns a Data object that has been augmented by
    `setup_spectral_data`.
    """
    
    w, U, x, y = load_from_matlab(dataset, filename, max_num_ev)
    
    data = Data(
        x = torch.tensor(x, dtype=torch.float),
        y = torch.tensor(y, dtype=torch.long),
        num_classes = int(y.max())+1,
        name = dataset)
    
    setup_spectral_data(data, w, U, threshold=threshold)
    
    return data