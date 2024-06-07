
import numpy as np
from warnings import warn

import torch
from torch_geometric.data import Data, InMemoryDataset
        
def setup_spectral_data(data, w, U, threshold=1e-2, max_rank=None, check_zero_multiplicity=True):
    w = torch.as_tensor(w, dtype=torch.float).flatten()
    U = torch.as_tensor(U, dtype=torch.float)
    nonzero_mask = w > threshold
    
    data.zero_U = U[:, ~nonzero_mask]
    data.nonzero_U = U[:, nonzero_mask]
    data.nonzero_w = w[nonzero_mask]
    data.eigengap = data.nonzero_w.min().item() if nonzero_mask.sum() > 0 else 0.0
    
    # if check_zero_multiplicity:
    #     zero_mult = data.zero_U.shape[1]
    #     if zero_mult != 1:
    #         print("setup_spectral_data: Multiplicity of Laplacian eigenvalue 0 is {} instead of expected 1".format(zero_mult))
    
    data.nonzero_w, ind = torch.sort(data.nonzero_w)
    data.nonzero_U = data.nonzero_U[:, ind]
    
    if max_rank is not None and data.nonzero_w.numel() > max_rank:
        data.nonzero_w = data.nonzero_w[:max_rank]
        data.nonzero_U = data.nonzero_U[:, :max_rank]
    data.rank = data.nonzero_w.numel()
    
    return data



def random_split(data, split_size=None):
    device = data.y.device
    y = data.y.detach().cpu().numpy()
    data.train_mask = torch.zeros(data.num_nodes, dtype=bool, device=device)
    for c in range(data.num_classes):
        ind = np.nonzero(y == c)[0]
        ind = np.random.choice(ind, split_size, replace=False)
        data.train_mask[ind] = True
    data.test_mask = ~data.train_mask


def fixed_split(data, index=0):
    assert "train_index_sets" in data, \
        ValueError("pinvgcn.data.fixed_split is only usable for data with train_index_sets field")
    data.train_mask = None
    data.train_idx = data.train_index_sets[index % data.train_index_sets.shape[0]]
    check_masks(data)


def check_masks(data):
    if 'train_mask' in data:
        if 'val_mask' not in data:
            data.val_mask = torch.zeros(data.num_nodes, dtype=bool)

        if 'test_mask' not in data:
            data.test_mask = ~data.train_mask & ~data.val_mask

    elif 'train_idx' in data:
        def idx_to_mask(idx):
            mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            mask[idx] = True
            return mask

        data.train_mask = idx_to_mask(data.train_idx)

        if 'val_idx' in data:
            data.val_mask = idx_to_mask(data.val_idx)
        else:
            data.val_mask = torch.zeros(data.num_nodes, dtype=bool)

        if 'test_idx' in data:
            data.test_mask = idx_to_mask(data.test_idx)
        else:
            data.test_mask = ~data.train_mask & ~data.val_mask

        data.train_idx = data.test_idx = data.val_idx = None
    else:
        data.train_mask = data.test_mask = data.val_mask = None
        

class SingleSliceDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.data.num_classes = self.num_classes
    
    @property
    def processed_file_names(self):
        return 'data.pt'

    def save_processed(self, **kwargs):
        data = Data(**kwargs)
        
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])
