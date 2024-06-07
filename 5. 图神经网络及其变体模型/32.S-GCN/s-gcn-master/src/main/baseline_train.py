import argparse
import os
import sys


def get_experiment_description(train_args):
    return [
        'id: {}'.format(train_args.id),
        'features: {}'.format(train_args.features),
        'network: {}'.format(train_args.network),
        'conv_nonlinearity: {}'.format(train_args.conv_nonlinearity),
        '-',
        'train_datasets: {}'.format(train_args.train_datasets),
        'include_near_native: {}'.format(train_args.include_near_native),
        'normalize_x: {}'.format(train_args.normalize_x),
        'normalize_adj: {}'.format(train_args.normalize_adj),
        'shuffle: {}'.format(train_args.shuffle),
        'threads: {}'.format(train_args.threads),
        'bad_targets: {}'.format(train_args.bad_targets is not None),
        'encoder: {}'.format(train_args.encoder),
        'message_passing: {}'.format(train_args.message_passing),
        'scorer: {}'.format(train_args.scorer),
        '-',
        'optim: {}'.format(train_args.optim),
        'lr: {}'.format(train_args.lr),
        'l2_reg: {}'.format(train_args.l2_reg),
        'dropout: {}'.format(train_args.dropout),
        'loss: {}'.format(train_args.loss),
        'train_size: {}'.format(train_args.train_size)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', action="store", type=str, required=True)
    parser.add_argument('--features', action="store", type=int, required=False, default=3)
    parser.add_argument('--network', action="store", type=str, required=True)
    parser.add_argument('--conv_nonlinearity', action="store", type=str, required=False, default='elu')

    parser.add_argument('--train_datasets', action="store", type=str, required=True)
    parser.add_argument('--train_data_path', action="store", type=str, required=True)
    parser.add_argument('--train_gemme_features_path', action="store", type=str, required=False, default=None)

    parser.add_argument('--atom_types_path', action="store", type=str, required=True)
    parser.add_argument('--include_near_native', action="store_true", default=False)
    parser.add_argument('--normalize_x', action="store_true", default=False)
    parser.add_argument('--normalize_adj', action="store_true", default=False)
    parser.add_argument('--res_seq_sep', action="store", type=int, required=False, default=1)
    parser.add_argument('--shuffle', action="store_true", default=False)
    parser.add_argument('--threads', action="store", type=int, required=False, default=1)
    parser.add_argument('--gpu', action="store", type=int, required=False, default=None)

    parser.add_argument('--optim', action="store", type=str, required=True)
    parser.add_argument('--lr', action="store", type=float, required=True)
    parser.add_argument('--l2_reg', action="store", type=float, required=False, default=0.0)
    parser.add_argument('--dropout', action="store", type=float, required=False, default=0.0)
    parser.add_argument('--loss', action="store", type=str, required=False, default='mse')
    parser.add_argument('--epochs', action="store", type=int, required=False, default=15)
    parser.add_argument('--train_size', action="store", type=int, required=False, default=None)
    parser.add_argument('--batch_size', action="store", type=int, required=False, default=64)
    parser.add_argument('--memory_size', action="store", type=int, required=False, default=512)
    parser.add_argument('--encoder', action="store", type=int, required=False, default=3)
    parser.add_argument('--message_passing', action="store", type=int, required=False, default=8)
    parser.add_argument('--scorer', action="store", type=int, required=False, default=3)

    parser.add_argument('--checkpoints', action="store", type=str, required=True)
    parser.add_argument('--log_path', action="store", type=str, required=False, default=None)
    parser.add_argument('--bad_targets', action="store", type=str, required=False, default=None)

    args = parser.parse_args()
    description = get_experiment_description(args)
    utils.output('\n'.join(description))
    output_checkpoint_path, last_checkpoint_path = utils.get_checkpoint(args.checkpoints, args.id)

    if args.log_path:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(args.log_path, args.id))
    else:
        writer = None

    dataloaders = data.get_dataloaders(
        datasets=utils.parse_datasets(args.train_datasets),
        data_path=args.train_data_path,
        atom_types_path=args.atom_types_path,
        include_near_native=args.include_near_native,
        normalize_x=args.normalize_x,
        normalize_adj=args.normalize_adj,
        res_seq_sep=args.res_seq_sep,
        shuffle=args.shuffle,
        bad_targets_path=args.bad_targets,
        number=args.threads,
        gpu=args.gpu)
    network = networks.get_network(
        name=args.network,
        features_dim=args.features,
        n_channels_r=args.res_seq_sep,
        enc_layers=args.encoder,
        conv_layers=args.message_passing,
        scoring_layers=args.scorer,
        conv_nonlinearity=args.conv_nonlinearity,
        dropout=args.dropout)
    training.train(
        network=network.cuda(args.gpu) if args.gpu is not None else network,
        dataloaders=dataloaders,
        loss_function=training.get_loss_function(args.loss),
        optimizer=training.get_optim(args.optim, network, args.lr, args.l2_reg),
        epochs=args.epochs,
        train_size=args.train_size,
        description=description,
        output_checkpoint_path=output_checkpoint_path,
        load_checkpoint_path=last_checkpoint_path,
        logger=writer,
        worker=workers.train_worker,
        worker_kwargs={
            'batch_size': args.batch_size}
    )


if __name__ == '__main__':
    sys.path.append(os.environ['MG_LEARNING_PATH'])
    from src.common import training, utils
    from src.baseline import data, networks, workers
    main()
