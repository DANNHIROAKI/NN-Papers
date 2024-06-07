
import numpy as np
import scipy.sparse as sp
import os
from queue import Queue
from warnings import warn

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid

from .data import setup_spectral_data, check_masks


def load_graph_data(name, dir, lcc=False):
    r"""Load a graph dataset and return its Data object."""
    
    full_name = name + '_LCC' if lcc else name
    path = os.path.join(dir, full_name)
    
    if name in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = Planetoid(path, name, pre_transform=GraphPreprocess(lcc))
        data = dataset.data
        data.num_classes = dataset.num_classes
    else:
        raise ValueError("Unknown dataset: {}".format(name))
    
    data.name = full_name
    return data


class GraphPreprocess(object):
    r"""Class in the style of a torch_geometric transform. Used internally in
    load_graph_data to preprocess the data and optionally compute its largest
    connected component."""
    
    def __init__(self, lcc=False):
        self.lcc = lcc
    
    def __call__(self, data):
        
        check_masks(data)
        
        if 'edge_weight' not in data:
            data.edge_weight = data.edge_attr if 'edge_attr' in data else None
            
        if self.lcc:
            data = lcc_data(data)
            
        return data

class GraphSpectralSetup(object):
    r"""
    Class in the style of a torch_geometric transform. Augments a data 
    object with spectral information on the graph Laplacian. If `rank` is not 
    None, a low-rank approximation is used. eig_tol is the tolerance for the 
    eigenvalue computation. `eig_threshold` determines which eigenvalues are 
    treated as zero. If loop_weights is not None, additional self loops are 
    added with that weight. If dense_graph is True, internal computations are 
    done with the adjacency matrix stored as a numpy array instead of a 
    scipy.sparse.coo_matrix.
    """
    
    
    def __init__(self, rank=None, loop_weights=None, dense_graph=False, eig_tol=0, eig_threshold=1e-6, precompute_U0=False):
        self.rank = rank
        self.loop_weights = loop_weights
        self.dense_graph = dense_graph
        self.eig_tol = eig_tol
        self.eig_threshold = eig_threshold
        self.precompute_U0 = precompute_U0
        
    def __call__(self, data):
        adj = normalized_adjacency(data, self.loop_weights, self.dense_graph)
        
        if self.rank is None:
            raise NotImplementedError("Full non-approximated Pseudoinverse is not available for graphs")
        else:
            U0 = precompute_U_zero(data) if self.precompute_U0 else None
            num_ev = self.rank + (U0.shape[1] if self.precompute_U0 else 1);
            
            w, U = graph_laplacian_decomposition(adj, num_ev, tol=self.eig_tol, precomputed_U0 = U0)
            setup_spectral_data(data, w, U, threshold=self.eig_threshold, max_rank = self.rank)
            
            if data.rank != self.rank:
                warn('Computed rank {} does not match target rank {}'.format(data.rank, self.rank))
            
        return data
    
class SBMData(Data):
    r"""Data subclass for generation of Stochastic Blockmodel data. Creates b
    samples for each of c classes. The first s samples of each class are 
    training samples. Graph edges are not created until generate_adjacency is
    called."""
    def __init__(self, p, q, c, b, s, **kwargs):
        y = np.hstack([i*np.ones(b, dtype=int) for i in range(c)])
        train_mask = np.zeros(c*b, dtype=bool)
        for i in range(c):
            train_mask[i*b:i*b+s] = True
        
        super().__init__(
            p=p, q=q, num_classes=c, block_size=b, split_size=s, 
            y = torch.tensor(y),
            train_mask = torch.tensor(train_mask),
            test_mask =  torch.tensor(~train_mask),
            __num_nodes__ = c*b,
            name = 'SBM-p{}-q{}-c{}-b{}-s{}'.format(p, q, c, b, s),
            **kwargs)
        
    def generate_adjacency(self):
        r""" Create random undirected, unweighted edges based on the p and q
        parameters given to the Data constructor."""
        c = self.num_classes
        b = self.block_size
        
        edges = []
        for c1 in range(c):
            for i in range(c1*b, (c1+1)*b):
                for j in range(i+1, (c1+1)*b):
                    if np.random.rand() < self.p:
                        edges.append([i,j])
                        edges.append([j,i])
                for c2 in range(c1+1, c):
                    for j in range(c2*b, (c2+1)*b):
                        if np.random.rand() < self.q:
                            edges.append([i,j])
                            edges.append([j,i])
                            
        self.edge_index = torch.tensor(np.array(edges).T, device=self.y.device)
        

def lcc_data(data):
    edge_index = data.edge_index.cpu().numpy()
    
    num_components, component_ind = connected_components(edge_index, data.num_nodes)
    
    if num_components:
        print('Largest connected component: Full graph')
        return data
    
    node_mask = component_ind == np.argmax([(component_ind == i).sum() for i in range(num_components)])
    print('Largest connected component: {}/{} nodes'.format(node_mask.sum(), data.num_nodes))

    node_indices = -np.ones(data.num_nodes, dtype=int)
    node_indices[node_mask] = np.arange(node_mask.sum())
    edge_mask = node_mask[edge_index[0]]
    edge_index = node_indices[edge_index[:,edge_mask]]

    kwargs = {}
    for key, val in data.__dict__.items():
        if val is None:
            pass
        elif key == 'x':
            val = val[node_mask,:]
        elif key in ['y','train_mask','test_mask','val_mask']:
            val = val[node_mask]
        elif key == 'edge_index':
            val = torch.tensor(edge_index)
        elif key in ['edge_weight', 'edge_attr']:
            val = val[edge_mask]
        kwargs[key] = val
    return Data(**kwargs)

def connected_components(edge_index, num_nodes):
    edge_index = edge_index[:, np.argsort(edge_index[0])]
    neighborhoods = np.zeros(num_nodes+1, dtype=int)
    j = 0
    for e in range(edge_index.shape[1]):
        i = edge_index[0, e]
        while j < i:
            j += 1
            neighborhoods[j] = e
    while j < num_nodes:
        j += 1
        neighborhoods[j] = edge_index.shape[1]
    # now the edge indices e with edge_index[0,e] == j are exactly those in range(neighborhoods[j], neighborhoods[j+1])
    
    components = np.zeros(num_nodes, dtype=int)
    c = 0
    q = Queue()
    for start in range(num_nodes):
        if components[start] != 0:
            continue
        c += 1
        components[start] = c
        q.put(start)
        while not q.empty():
            j = q.get()
            for k in edge_index[1, neighborhoods[j]:neighborhoods[j+1]]:
                if components[k] == 0:
                    components[k] = c
                    q.put(k)

    return c, components-1


def adjacency(data, loop_weights=None, dense=False):
    r"""Returns the adjacency matrix resulting from the edges given by the
    edge_index (and optionally edge_weight) fields in the data object. If 
    loop_weights is not None, additional self loops are added with that
    weight. By default, a scipy.sparse.coo_matrix is returned. If dense is 
    True, it is converted into a numpy array instead."""
    n = data.num_nodes
    ii, jj = data.edge_index.cpu().numpy()
    if 'edge_weight' not in data:
        ww = np.ones(data.num_edges, dtype=np.float32)
    else:
        ww = data.edge_weight.cpu().numpy()

    if loop_weights is not None and loop_weights != 0:
        ii = np.append(ii, np.arange(n))
        jj = np.append(jj, np.arange(n))
        ww = np.append(ww, loop_weights*np.ones(n))
    adj = sp.coo_matrix((ww, (ii,jj)), shape=(n,n))
    
    if dense:
        adj = adj.toarray()
    return adj

def normalized_adjacency(data, loop_weights=None, dense=False):
    r"""Returns the symmetrically normalized adjacency matrix resulting from 
    the edges given by the edge_index (and optionally edge_weight) fields in 
    the data object. The result is the unnormalized adjacency matrix 
    multiplied with the diagonal inverse square root degree matrix from both
    sides. If loop_weights is not None, additional self loops are added with 
    that weight. By default, a scipy.sparse.coo_matrix is returned. If dense is 
    True, it is converted into a numpy array instead.
    """
    adj = adjacency(data, loop_weights, dense)
    d = np.squeeze(np.asarray(adj.sum(1)))
    d = 1/np.sqrt(d)
    if dense:
        return d[:,np.newaxis] * adj * d
    else:
        return adj.multiply(d).multiply(d[:,np.newaxis]).tocsr()

def precompute_U_zero(data):
    if 'edge_index' not in data:
        return None
    n = data.num_nodes
    num_components, component_ind = connected_components(data.edge_index.cpu().numpy(), n)
    
    U0 = np.array([(component_ind == i).astype(np.float64) for i in range(num_components)]).T
    return U0 / np.sqrt(U0.sum(0));
                
    
def graph_laplacian_decomposition(adj, num_ev=None, tol=0, precomputed_U0=None):
    r"""Return a (partial) eigen decomposition of the graph Laplacian. If
    num_ev is not None, only that many smallest eigenvalues are computed. The 
    parameter tol is used for scipy.linalg.eigs (if it is called)."""
    n = adj.shape[0]
    if num_ev is None or num_ev > n/2:
        if sp.issparse(adj):
            adj = adj.toarray()
        w, U = np.linalg.eigh(adj)
        w = 1-w
        ind = np.argsort(w)
        if num_ev is not None:
            ind = ind[:num_ev]
        w = w[ind]
        U = U[:,ind]
    else:
        if precomputed_U0 is not None:
            matvec = lambda x: adj @ x + x - 2*precomputed_U0 @ (precomputed_U0.T @ x)
            shifted_adj = sp.linalg.LinearOperator((n,n), matvec)
            num_ev -= precomputed_U0.shape[1]
        elif sp.issparse(adj):
            shifted_adj = (adj + sp.identity(adj.shape[0])).tocsr()
        else:
            shifted_adj = adj + np.identity(adj.shape[0])
            
        w, U = sp.linalg.eigsh(shifted_adj, num_ev, tol=tol)
        w = 2-w
        
        if precomputed_U0 is not None:
            # print("Eigenvalue computation with precomputed U0: zero multiplicity {}, eigengap {:.4f}".format(
            #     precomputed_U0.shape[1], w.min()))
            U = np.hstack((precomputed_U0, U))
            w = np.hstack((np.zeros(precomputed_U0.shape[1]), w))
        
    return w.astype(np.float32), U.astype(np.float32)
