import numpy as np
import pandas as pd
import time
import torch

from src.common import names, utils
from tqdm import tqdm


def split(a, n):
    k, m = divmod(len(a), n)
    return {(i + 1) * k + min(i + 1, m) - 1 for i in range(n)}


def train_worker(network, loader, optimizer, loss_function, size, p_id, p_outputs, batch_size=64, memsize=512):
    train_loss = []
    predictions = []
    ground_truth = []
    start_time = time.time()

    # utils.output('Started training process {}'.format(p_id))
    network.train()
    if p_id == 0:
        generator = tqdm(loader.generate(size=size, memsize=memsize, training=True), total=size)
    else:
        generator = loader.generate(size=size, memsize=memsize, training=True)
    borders = split(range(size), size // batch_size)
    for it, graph in enumerate(generator):

        model_sequence = graph[names.MODEL_SEQUENCE_FIELD]
        target_sequence = graph[names.TARGET_SEQUENCE_FIELD]

        one_hot = graph[names.ONE_HOT_FIELD]
        features = graph[names.FEATURES_FIELD]
        y = graph[names.Y_FIELD]
        sh = graph[names.SH_FIELD]

        pred = network(one_hot, features, sh).squeeze()
        assert not torch.any(torch.isnan(pred))

        # aligning local_predictions with y
        if y is not None and len(pred) < len(y):
            left = pd.DataFrame({'residue': model_sequence})
            right = pd.DataFrame({'residue': target_sequence, 'idx': np.array(len(target_sequence))})
            merged = left.merge(right, on='residue')
            y = y[merged.idx.values]

        loss = loss_function(pred, y)
        assert not torch.any(torch.isnan(loss))

        loss.backward()
        if it in borders:
            optimizer.step()
            optimizer.zero_grad()

        loss_np = loss.cpu().detach().numpy()
        pred_np = pred.cpu().detach().numpy()
        gt_np = y.cpu().detach().numpy()
        train_loss.append(loss_np)
        predictions.append(pred_np.mean())
        ground_truth.append(gt_np.mean())

    p_outputs[p_id] = {
        'train_loss': train_loss,
        'predictions': predictions,
        'ground_truth': ground_truth,
        'time': time.time() - start_time
    }
    # utils.output('Finished training process {}'.format(p_id))


def predict_single_graph(graph, network):
    network.eval()
    one_hot = graph[names.ONE_HOT_FIELD]
    features = graph[names.FEATURES_FIELD]
    sh = graph[names.SH_FIELD]
    local_predictions = network(one_hot, features, sh).squeeze().cpu().detach().numpy()
    global_prediction = np.mean(local_predictions)
    y = graph[names.Y_FIELD] if graph[names.Y_FIELD] is not None else local_predictions
    y_global = graph[names.Y_GLOBAL_FIELD] if graph[names.Y_GLOBAL_FIELD] is not None else global_prediction
    return local_predictions, global_prediction, y, y_global


def predict_and_save_worker(network, loader, output_path, p_id):
    # utils.output('Started prediction process {}'.format(p_id))
    if p_id == 0:
        generator = tqdm(loader.generate(), total=len(loader.models_info))
    else:
        generator = loader.generate()
    for graph in generator:
        target_name = graph[names.TARGET_NAME_FIELD]
        model_name = graph[names.MODEL_NAME_FIELD]
        local_predictions, global_prediction, y, y_global = predict_single_graph(graph, network)
        pred_df = pd.DataFrame({'pred': local_predictions, 'y': y})
        pred_df.to_csv(utils.path([output_path, target_name, model_name + '.csv']), index=False)
        with open(utils.path([output_path, target_name, model_name + '.y']), 'w') as f:
            f.write(str(y_global))
    # utils.output('Process {} finished'.format(p_id))
