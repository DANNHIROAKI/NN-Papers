import numpy as np
import os
import pandas as pd
import torch
import torch.sparse as sparse

from src.common import utils, names

CORRECT_MODEL_FILES_SET = {
    names.MODEL_FILE_NAME,
    names.X_RES_FILE_NAME,
    names.ADJ_RES_NAME,
}


def get_dataloader_description(
        datasets,
        targets,
        shuffle,
        include_near_native,
        normalize_adj,
        normalize_x,
        res_seq_sep,):

    descr = ""
    descr += "Data:\n"
    descr += "-- datasets: {}\n".format(datasets)
    descr += "-- targets: {}\n".format(len(targets))
    descr += "-- include_near_native: {}\n".format(include_near_native)
    descr += "-- shuffle: {}\n".format(shuffle)
    descr += "Parameters:\n"
    descr += "-- normalize_adj: {}\n".format(normalize_adj)
    descr += "-- normalize_x: {}\n".format(normalize_x)
    descr += "-- res_seq_sep: {}\n".format(res_seq_sep)
    return descr


def get_dataloaders(
        datasets,
        data_path,
        atom_types_path,
        include_near_native,
        normalize_adj,
        normalize_x,
        res_seq_sep,
        shuffle,
        bad_targets_path=None,
        gemme_features_path=None,
        number=1,
        gpu=None):

    print(' ')
    utils.output('Getting dataloaders...')

    if bad_targets_path is not None:
        bad_targets = utils.get_bad_targets(bad_targets_path)
    else:
        bad_targets = set()
    utils.output('Loaded {} bad targets'.format(len(bad_targets)))

    target_to_gemme_path = dict()
    if gemme_features_path is not None:
        for dataset_name in datasets:
            for gemme_features_file in os.listdir(utils.path([gemme_features_path, dataset_name])):
                target = gemme_features_file.split('.')[0]
                target_to_gemme_path[target] = utils.path([gemme_features_path, dataset_name, gemme_features_file])

    target_to_path = dict()
    targets = []
    for dataset_name in datasets:
        for target_name, target_path in utils.iterate_targets(utils.path([data_path, dataset_name]), bad_targets):
            if gemme_features_path is None or target_name in target_to_gemme_path.keys():
                target_to_path[target_name] = target_path
                targets.append(target_name)

    targets = sorted(targets)

    residue_type_to_id = utils.get_residue_type_to_id(atom_types_path)
    description = get_dataloader_description(
        datasets=datasets,
        targets=targets,
        shuffle=shuffle,
        include_near_native=include_near_native,
        normalize_adj=normalize_adj,
        normalize_x=normalize_x,
        res_seq_sep=res_seq_sep)
    utils.output(description)

    i = 0
    model_info = [[] for _ in range(number)]
    for target_name in targets:
        gemme_path = target_to_gemme_path[target_name] if gemme_features_path is not None else None
        for model_name, model_path in utils.iterate_models(
                target_path=target_to_path[target_name],
                include_near_native=include_near_native,
                correct_model_files_set=CORRECT_MODEL_FILES_SET):

            model_info[i % number].append({
                'target_name': target_name,
                'model_name': model_name,
                'model_path': model_path,
                'gemme_path': gemme_path,
            })
            i += 1

    dataloaders = []
    for i in range(number):
        curr_models_info = model_info[i]
        if shuffle:
            np.random.shuffle(curr_models_info)
        loader = LightDataLoader(
            models_info=curr_models_info,
            residue_type_to_id=residue_type_to_id,
            shuffle=shuffle,
            include_near_native=include_near_native,
            normalize_adj=normalize_adj,
            normalize_x=normalize_x,
            res_seq_sep=res_seq_sep,
            description=description,
            gpu=gpu)
        dataloaders.append(loader)

    return dataloaders


class LightDataLoader:
    def __init__(self, models_info, residue_type_to_id, shuffle, include_near_native,
                 normalize_adj, normalize_x, res_seq_sep, description, gpu):

        self.models_info = models_info
        self.residue_type_to_id = residue_type_to_id
        self.shuffle = shuffle

        self.include_near_native = include_near_native
        self.normalize_adj = normalize_adj
        self.normalize_x = normalize_x
        self.res_seq_sep = res_seq_sep
        self.description = description

        self.start_pos = 0
        self.gpu = gpu

    def generate(self, size=None, training=False):
        if size is None:
            curr_models_info = self.models_info
        elif self.start_pos + size < len(self.models_info):
            curr_models_info = self.models_info[self.start_pos:self.start_pos + size]
            self.start_pos += size
        else:
            second_part_size = size - (len(self.models_info) - self.start_pos)
            curr_models_info = self.models_info[self.start_pos:] + self.models_info[:second_part_size]
            self.start_pos = second_part_size
        for model_info in curr_models_info:
            target_name = model_info['target_name']
            model_name = model_info['model_name']
            model_path = model_info['model_path']
            gemme_path = model_info['gemme_path']
            graph = self.build_graph(
                model_name=model_name,
                model_path=model_path,
                target_name=target_name,
                gemme_path=gemme_path)
            if graph['y'] is not None or training is False:
                yield graph

    def build_graph(self, model_name, model_path, target_name, gemme_path):
        one_hot, features, gemme_features = self.load_x(utils.path([model_path, names.X_RES_FILE_NAME]), gemme_path)

        # y can not exist for testing data
        y_path = utils.path([model_path, names.Y_FILE_NAME])
        y = pd.read_csv(y_path, sep=' ')['score'].values if os.path.exists(y_path) else None

        # y_global can not exist for testing data
        y_global_path = utils.path([model_path, names.Y_GLOBAL_FILE_NAME])
        y_global = np.array([float(open(y_global_path).read().strip())]) if os.path.exists(y_global_path) else None

        adj_res_sparse_parts = self.load_res_adj(
            utils.path([model_path, names.ADJ_RES_NAME]),
            len(one_hot))

        if self.gpu is None:
            return {
                'target_name': target_name,
                'model_name': model_name,
                'one_hot': torch.from_numpy(one_hot).type(torch.float32),
                'features': torch.from_numpy(features).type(torch.float32),
                'gemme_features': torch.from_numpy(gemme_features).type(torch.float32),
                'y': torch.from_numpy(y).type(torch.float32) if y is not None else None,
                'y_global': torch.from_numpy(y_global).type(torch.float32) if y_global is not None else None,
                'adj_res': list(map(lambda x: sparse.FloatTensor(*x), adj_res_sparse_parts)),
            }
        else:
            return {
                'target_name': target_name,
                'model_name': model_name,
                'one_hot': torch.from_numpy(one_hot).type(torch.float32).cuda(self.gpu),
                'features': torch.from_numpy(features).type(torch.float32).cuda(self.gpu),
                'gemme_features': torch.from_numpy(gemme_features).type(torch.float32).cuda(self.gpu),
                'y': torch.from_numpy(y).type(torch.float32).cuda(self.gpu) if y is not None else None,
                'y_global':
                    torch.from_numpy(y_global).type(torch.float32).cuda(self.gpu) if y_global is not None else None,
                'adj_res': list(map(lambda x: sparse.FloatTensor(*x).cuda(self.gpu), adj_res_sparse_parts)),
            }

    def load_x(self, file, gemme_path):
        compressed_x = pd.read_csv(file, sep=' ')
        residue_types = compressed_x['residue'].values
        features = compressed_x[['volume', 'buriedness', 'sasa']].values.astype(np.float32)

        if self.normalize_x:
            features_to_normalize = features[:, names.X_NORM_IDX]
            normalizations = 1 / features_to_normalize.sum(axis=0)
            normalized_features = np.einsum('ij,j->ij', features_to_normalize, normalizations)
            features[:, names.X_NORM_IDX] = normalized_features

        one_hot = np.zeros((len(residue_types), len(self.residue_type_to_id)))
        for i, residue_type in enumerate(residue_types):
            one_hot[i, self.residue_type_to_id[residue_type]] = 1

        gemme_features = np.array([]) if gemme_path is None else np.genfromtxt(gemme_path).T
        return one_hot, features, gemme_features

    def get_normalized_values(self, edges):
        if self.normalize_adj:
            sums = np.bincount(edges[:, 0].astype(int), weights=edges[:, 2])
            return np.array([edge[2] / (sums[int(edge[0])] + names.EPSILON) for edge in edges])
        else:
            return edges[:, 2]

    def load_res_adj(self, edges_file, n):
        edges = np.genfromtxt(edges_file)
        edges = np.concatenate([edges, np.stack([edges[:, 1], edges[:, 0], edges[:, 2]]).T])
        edges[:, 2] = self.get_normalized_values(edges)
        if self.res_seq_sep == 1:
            edges_ind_t = torch.LongTensor(edges[:, :2].T)
            edges_val_t = torch.FloatTensor(edges[:, 2])
            return [(edges_ind_t, edges_val_t, torch.Size([n, n]))]
        else:
            inds_by_type = [[[], []] for i in range(self.res_seq_sep)]
            vals_by_type = [[] for i in range(self.res_seq_sep)]
            for i, j, val in edges:
                if abs(i - j) >= self.res_seq_sep:
                    inds_by_type[-1][0].append(i)
                    inds_by_type[-1][1].append(j)
                    vals_by_type[-1].append(val)
                else:
                    inds_by_type[self.res_seq_sep - 1][0].append(i)
                    inds_by_type[self.res_seq_sep - 1][1].append(j)
                    vals_by_type[self.res_seq_sep - 1].append(val)
            return [
                (torch.LongTensor(inds), torch.FloatTensor(vals), torch.Size([n, n]))
                for inds, vals in zip(inds_by_type, vals_by_type)
            ]

    def describe(self):
        return self.description
