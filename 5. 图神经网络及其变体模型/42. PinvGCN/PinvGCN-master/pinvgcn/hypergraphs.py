
import numpy as np
import scipy.sparse as sp
import os
import shutil
import pickle

import torch
from torch_geometric.data import download_url
from urllib.error import HTTPError

from .data import setup_spectral_data, SingleSliceDataset


def load_hypergraph_data(name, dir, categorical_regularization = 0):
    r"""Load a hypergraph dataset and return its Data object. Currently
    supported dataset names are Mushroom, Covertype45, and Covertype67. Upon
    first usage, data is downloaded from the UCI website and then processed to
    turn the categorial attributes into hyperedges."""
    
    path = os.path.join(dir, name)
    
    if name == 'Mushroom':
        data = MushroomDataset(path).data
    elif name == 'Covertype45':
        data = CovertypeDataset(path, [4,5]).data
    elif name == 'Covertype67':
        data = CovertypeDataset(path, [6,7]).data
    elif name.endswith("-Coauthorship"):
        data = CitationHypergraphDataset(path, "Coauthorship", name[:-13]).data
    elif name.endswith("-Cocitation"):
        data = CitationHypergraphDataset(path, "Cocitation", name[:-11]).data
    else:
        raise ValueError('Unknown hypergraph dataset name: ' + name)
    
    data.name = name
    
    if categorical_regularization > 0:
        regularize_with_categorical_features(data, categorical_regularization)
    
    return data



class HypergraphSpectralSetup(object):
    r"""
    Class in the style of a torch_geometric transform. Augments a data object
    with spectral information on the hypergraph Laplacian. If `rank` is not 
    None, a low-rank approximation is used. If `partial_eigs` is True,
    only the relevant eigenvalues are computed, which might negatively impact
    runtime for small systems. In the partial case, `eig_tol` is the tolerance
    for the eigenvalue computation. `eig_threshold` determines which 
    eigenvalues are treated as zero.
    This function currently expects the hypergraph incidence to be stored in
    data.x as a dense tensor.
    """
    
    def __init__(self, rank=None, eig_tol=0, eig_threshold=1e-6, partial_eigs=False):
        self.rank = rank
        self.eig_tol = eig_tol
        self.eig_threshold = eig_threshold
        self.partial_eigs = partial_eigs
        
    def __call__(self, data):
        inc = normalized_incidence(data, sparse=self.partial_eigs)
    
        w, U = hypergraph_laplacian_decomposition(inc, num_ev=None if self.rank is None else self.rank+1, 
                                                  partial_svd=self.partial_eigs, partial_tol=self.eig_tol)
    
        setup_spectral_data(data, w, U, threshold=self.eig_threshold, max_rank=self.rank)
    
        return data


class HypergraphDataset(SingleSliceDataset):
    def save_hypergraph(self, x, y, hyperedge_index=None, hyperedge_weight=None, **kwargs):
        self.save_processed(
            x = torch.as_tensor(x, dtype=torch.float),
            y = torch.as_tensor(y, dtype=torch.long),
            hyperedge_index = None if hyperedge_index is None else \
                torch.as_tensor(hyperedge_index, dtype=torch.long),
            hyperedge_weight = None if hyperedge_weight is None else \
                torch.as_tensor(hyperedge_weight, dtype=torch.float),
            **{k: torch.as_tensor(v) for k, v in kwargs.items()})
        
class MushroomDataset(HypergraphDataset):
    r"""Subclass of InMemoryDataset that downloads and processes the Mushroom
    dataset from the UCI website."""
    
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data'
    
    @property
    def raw_file_names(self):
        return 'agaricus-lepiota.data'
    
    def download(self):
        download_url(self.url, self.raw_dir)
        
    def process(self):
        raw = []
        with open(os.path.join(self.raw_dir, self.raw_file_names), 'r') as f:
            for line in f:
                raw.append(line[:-1].split(','))
                
        raw = np.array(raw)
        labels = (raw[:,0] == 'e').astype(int)
        
        inc_list = []
        for i in range(1, raw.shape[1]):
            col = raw[:,i]
            values = np.unique(col)
            if any([v.startswith('?') for v in values]):
                continue
            
            for v in values:
                edge = (col == v)
                print(' - Hyperedge size for attribute #{} == {}:  {}'.format(i, v, edge.sum()))
                inc_list.append(edge)
                
        incidence = np.array(inc_list, dtype=np.float).T
        print(' - Mushroom incidence shape: {}'.format(incidence.shape))
    
        self.save_hypergraph(incidence, labels)
        


class CovertypeDataset(HypergraphDataset):
    r"""Subclass of InMemoryDataset that downloads and processes the Covertype
    dataset from the UCI website. A subset of the data can be used by only 
    using certain classes."""
    
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz'
    
    def __init__(self, root, classes=None, num_bins=10, transform=None, pre_transform=None):
        self.classes = classes
        self.num_bins = num_bins
        super().__init__(root, transform, pre_transform)
    
    @property
    def raw_file_names(self):
        return 'covtype.data'
    
    def download(self):
        download_url(self.url, self.raw_dir)
        os.system('gunzip -f ' + os.path.join(self.raw_dir, self.raw_file_names + '.gz'))
        
    def process(self):
        raw_attributes, labels = [], []
        with open(os.path.join(self.raw_dir, self.raw_file_names), 'r') as f:
            for line in f:
                values = line.split(',')
                raw_attributes.append([int(v) for v in values[:-1]])
                labels.append(int(values[-1]))
        raw_attributes = np.array(raw_attributes)
        labels = np.array(labels)
        
        inc_list = []
        for attr in range(10):
            col = raw_attributes[:,attr]
            colmin = col.min()
            colmax = col.max()
            print(' - Continuous attribute #{}: min {}, max {}'.format(attr, colmin, colmax))
            
            b = colmin
            for i in range(self.num_bins):
                a = b
                b = colmax+1 if i==self.num_bins-1 else colmin + (colmax-colmin)*(i+1)/self.num_bins
                inc_list.append((col >= a) & (col < b))
        
        incidence = np.hstack((np.array(inc_list, dtype=int).T, raw_attributes[:,10:]))
        print(' - Full Covertype incidence shape: {}'.format(incidence.shape))
        
        if self.classes is not None:
            
            mask = [l in self.classes for l in labels]
            class_map = {orig: new for new, orig in enumerate(self.classes)}
            labels = [class_map[l] for l in labels[mask]]
            
            incidence = incidence[mask, :]
            edge_deg = incidence.sum(axis=0)
            mask = edge_deg > 0 # include hyperedges with a single node
            for attr in range(10):
                print(' - Hyperedges for continuous attribute #{}: {} out of {}'.format(attr, mask[attr*self.num_bins : (attr+1)*self.num_bins].sum(), self.num_bins))
            incidence = incidence[:, mask]
            
            print(' - Partial incidence shape for classes {}: {}'.format(self.classes, incidence.shape))
            
        self.save_hypergraph(incidence, labels)


class CitationHypergraphDataset(HypergraphDataset):
    
    def __init__(self, root, network_type="Coauthorship", dataset="Cora", transform=None, pre_transform=None):
        self.network_type = network_type.lower()
        self.dataset = dataset.lower()
        super().__init__(root, transform, pre_transform)
    
    @property
    def url_base(self):
        return 'https://github.com/malllabiisc/HyperGCN/raw/master/data/{}/{}/'.format(self.network_type, self.dataset)
    
    @property
    def raw_file_names(self):
        return ["features.pickle", "labels.pickle", "hypergraph.pickle"] + \
            ["split{}.pickle".format(i) for i in range(1,11)]
    
    def download(self):
        try:
            for file_name in self.raw_file_names:
                is_split = file_name.startswith("split")
                download_url(self.url_base + ("splits/" + file_name[5:] if is_split else file_name), self.raw_dir)
                if is_split:
                    os.replace(os.path.join(self.raw_dir, file_name[5:]),
                               os.path.join(self.raw_dir, file_name))
                    
        except HTTPError as e:
            if e.code == 404:
                # there likely was a typo, so remove the whole root
                shutil.rmtree(self.root) 
                raise ValueError("Did not find {} dataset {} at {}".format(
                    self.network_type, self.dataset, e.filename)) from e
            else:
                raise
            
    def process(self):
        
        def load(obj_name):
            with open(os.path.join(self.raw_dir, obj_name + ".pickle"), "rb") as f:
                return pickle.load(f)
        
        features = load("features").toarray()
        labels = np.array(load("labels"))
        
        hyperedge_dict = load("hypergraph")
        hyperedge_index = np.hstack([np.vstack(([e] * len(nodes), list(nodes)))
                                     for e, nodes in enumerate(hyperedge_dict.values())])
        num_hyperedges = len(hyperedge_dict)
        hyperedge_weight = np.ones(num_hyperedges)
        
        train_index_sets = torch.tensor([load("split{}".format(i))['train'] for i in range(1,11)],
                                        dtype=torch.long)
        
        self.save_hypergraph(features, labels, hyperedge_index, hyperedge_weight,
                             train_index_sets = train_index_sets)

def regularize_with_categorical_features(data, weight=1):
    
    assert 'hyperedge_index' in data, ValueError("Regularization with categorical features currently only works if the hypergraph is given in sparse form")
    
    parts = [data.hyperedge_index]
    num_hyperedges = data.hyperedge_index[0].max().item() + 1
    
    if 'hyperedge_weight' not in data and weight != 1:
        data.hyperedge_weight = torch.ones(num_hyperedges, dtype=torch.float)
    
    for i in range(data.x.shape[1]):
        node_indices = torch.nonzero(data.x[:,i], as_tuple=True)[0]
        if len(node_indices) > 1:
            parts.append(torch.cat((torch.tensor([[num_hyperedges]*len(node_indices)], dtype=torch.long),
                                    node_indices.reshape(1,-1)), dim=0))
            num_hyperedges += 1
    
    if len(parts) > 1:
        data.hyperedge_index = torch.cat(parts, dim=1)
        
        if 'hyperedge_weight' in data:
            data.hyperedge_weight = torch.cat((data.hyperedge_weight,
                                               weight*torch.ones(num_hyperedges - len(data.hyperedge_weight),
                                                                 dtype=torch.float)))
        

def normalized_incidence(data, sparse=False):
    r"""Compute a numpy array holding D^{-1/2} H W^{1/2} B^{-1/2}, where H is 
    the hypergraph incidence matrix. If data.hyperedge_index is given, it is
    
    given by data.x, D is the diagonal node
    degree matrix, B is the diagonal hyperedge degree matrix, and W is the
    optional diagonal hyperedge weight matrix given by data.hyperedge_weight.
    """
    
    if 'hyperedge_index' in data:
        col, row = data.hyperedge_index.cpu().numpy()
        inc = sp.coo_matrix((np.ones(row.size), (row,col)), shape=(data.num_nodes, col.max()+1))
        if not sparse:
            inc = inc.toarray()
    else:
        # assume that data.x contains the dense incidence matrix
        inc = data.x.cpu().numpy()
        if sparse:
            inc = sp.coo_matrix(inc)
        
    if 'hyperedge_weight' in data:
        weights = data.hyperedge_weight.cpu().numpy()
    else:
        weights = np.ones(inc.shape[1], dtype=np.float)
    
    d = inc @ weights
    d_mask = d > 1e-4
    d[d_mask] = 1/np.sqrt(d[d_mask])
    b = np.ones(inc.shape[0]) @ inc
    b = np.sqrt(weights / b)
    
    if sparse:
        for i in range(inc.nnz):
            inc.data[i] *= d[inc.row[i]] * b[inc.col[i]]
        return inc.tocsr()
    else:
        return d[:,np.newaxis] * inc * b

def hypergraph_laplacian_decomposition(inc, num_ev=None, partial_svd=False, partial_tol=0):
    r"""Return a (partial) eigen decomposition of the hypergraph Laplacian. If
    num_ev is not None, only that many smallest eigenvalues are computed. If 
    partial_svd is True, only a partial SVD is computed via scipy.linalg.svds
    with tolerance given by the parameter partial_tol."""
    
    # TODO: get better heuristic of when to choose SVDS over SVD
    if partial_svd and num_ev is not None:
        U, sigma, _ = sp.linalg.svds(inc, num_ev, tol=partial_tol)
    else:
        U, sigma, _ = np.linalg.svd(inc, full_matrices=False)
        if num_ev is not None:
            U = U[:,:num_ev]
            sigma = sigma[:num_ev]
        
    return 1 - sigma.astype(np.float32)**2, U.astype(np.float32)
