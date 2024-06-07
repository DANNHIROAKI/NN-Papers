import numpy as np
import torch
import torch.multiprocessing as mp

from src.common import names, utils


def get_optim(optim_name, network, lr, l2_reg=0):
    if optim_name == 'sgd':
        return torch.optim.SGD(network.parameters(), lr, weight_decay=l2_reg)
    elif optim_name == 'adam':
        return torch.optim.Adam(network.parameters(), lr, weight_decay=l2_reg)
    else:
        raise Exception('Invalid --optim argument')


def get_loss_function(loss_name):
    if loss_name == 'mse':
        return torch.nn.functional.mse_loss
    else:
        raise Exception('Invalid --loss argument')


def train(
        network,
        dataloaders,
        loss_function,
        optimizer,
        epochs,
        train_size,
        description,
        output_checkpoint_path,
        load_checkpoint_path,
        logger,
        worker,
        worker_kwargs=None):

    kwargs_str = ', '.join(['{}={}'.format(k, v) for k, v in worker_kwargs.items()]) \
        if worker_kwargs is not None else None
    utils.output('Training with epochs={}, threads={}, train_size={}, {}'.format(
        epochs, len(dataloaders), train_size, kwargs_str))
    if load_checkpoint_path is not None:
        state = torch.load(load_checkpoint_path)
        network.load_state_dict(state[names.MODEL_STATE])
        optimizer.load_state_dict(state[names.OPTIMIZER_STATE])
        train_loss_history = state[names.TRAIN_LOSS_HISTORY]
        train_corr_history = state[names.TRAIN_CORR_HISTORY]
        train_time_history = state[names.TRAIN_TIME_HISTORY]
        start_epoch = state[names.EPOCH] + 1
        utils.output('Starting from epoch: {}'.format(start_epoch))
    else:
        train_loss_history = []
        train_corr_history = []
        train_time_history = []
        start_epoch = 0

    mp.set_start_method('spawn')
    network.share_memory()

    for epoch in range(start_epoch, epochs):
        if epoch > 0:
            utils.output('Loss:')
            utils.output('--train: {}'.format(train_loss_history[-1]))
            if logger:
                logger.add_scalar("Loss/train", float(train_loss_history[-1]), epoch)
            utils.output('Corr:')
            utils.output('--train: {}'.format(train_corr_history[-1]))
            if logger:
                logger.add_scalar('Pearson/train', float(train_corr_history[-1]), epoch)

        utils.output('Epoch {}'.format(epoch))
        utils.output('Training...')

        manager = mp.Manager()
        p_outputs = manager.dict()
        processes = []
        for i, loader in enumerate(dataloaders):
            kwargs = {
                'network': network,
                'loader': loader,
                'optimizer': optimizer,
                'loss_function': loss_function,
                'size': train_size,
                'p_id': i,
                'p_outputs': p_outputs,
            }
            kwargs.update(worker_kwargs)
            p = mp.Process(target=worker, kwargs=kwargs)
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        epoch_loss = []
        epoch_predictions = []
        epoch_ground_truth = []
        epoch_timings = []
        for out in p_outputs.values():
            epoch_loss += out['train_loss']
            epoch_predictions += out['predictions']
            epoch_ground_truth += out['ground_truth']
            epoch_timings.append(out['time'])
        if len(epoch_loss) == 0:
            utils.output('Did not found any graph appropriate for training')
        else:
            train_loss_history.append(np.mean(epoch_loss))
            train_corr_history.append(np.corrcoef(epoch_predictions, epoch_ground_truth)[0, 1])
            train_time_history.append(np.mean(epoch_timings))

        utils.output('Saving checkpoint...')
        checkpoint = {
            names.MODEL_DESCRIPTION: '\n'.join(description),
            names.MODEL_STATE: network.state_dict(),
            names.OPTIMIZER_STATE: optimizer.state_dict(),
            names.TRAIN_LOSS_HISTORY: train_loss_history,
            names.TRAIN_CORR_HISTORY: train_corr_history,
            names.TRAIN_TIME_HISTORY: train_time_history,
            names.EPOCH: epoch,
        }
        utils.mkdir(output_checkpoint_path)
        chkp_epoch_path = utils.path([output_checkpoint_path, '{}_{}'.format(names.CHECKPOINT_EPOCH_PREFIX, epoch)])
        torch.save(checkpoint, chkp_epoch_path)
    if logger:
        logger.close()
