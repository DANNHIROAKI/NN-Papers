import argparse
import os
import sys
import torch
import warnings

from torch.multiprocessing import set_start_method

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict_only', action="store_true", default=False)
    parser.add_argument('--checkpoints', action="store", type=str, required=True)
    parser.add_argument('--dataset', action="store", type=str, required=True)
    parser.add_argument('--data', action="store", type=str, required=True)
    parser.add_argument('--atom_types_path', action="store", type=str, required=True)
    parser.add_argument('--prediction_output', action="store", type=str, required=True)
    parser.add_argument('--evaluation_output', action="store", type=str, required=True)
    parser.add_argument('--gpu', action="store", type=int, required=False, default=None)
    parser.add_argument('--threads', action="store", type=int, required=True)
    parser.add_argument('--epochs', action="store", type=str, required=False, default=None)
    args = parser.parse_args()

    model_id = args.checkpoints.split('/')[-1]
    prediction_output = utils.path([args.prediction_output, model_id, args.dataset])
    evaluation_directory = utils.path([args.evaluation_output, args.dataset])
    evaluation_output = utils.path([args.evaluation_output, args.dataset, model_id + '.eval'])
    utils.mkdir(evaluation_directory)
    if not os.path.exists(evaluation_output) and not args.predict_only:
        with open(evaluation_output, 'w') as f:
            f.write('epoch,' + ','.join(evaluation.ALL_METRICS_NAMES) + '\n')
    utils.mkdir(prediction_output)
    utils.mkdir(evaluation_output)
    loaders = None

    set_start_method('spawn')
    for checkpoint in utils.get_epochs(args):
        state = torch.load(utils.path([args.checkpoints, checkpoint]), map_location=torch.device('cpu'))
        params_dict = utils.parse_experiment_description(state['model_description'])
        current_prediction_output = utils.path([prediction_output, checkpoint])
        if not utils.predictions_are_computed(checkpoint, prediction_output):
            utils.mkdir(current_prediction_output)
            network = networks.get_network(
                name=params_dict['network'],
                features_dim=int(params_dict['features']),
                order=int(params_dict['order']),
                conv_nonlinearity=params_dict['conv_nonlinearity'],
                dropout=float(params_dict['dropout']))
            if args.gpu:
                network = network.cuda(args.gpu)
            network.load_state_dict(state['model_state'])
            network.eval()
            if loaders is None:
                loaders = data.get_dataloaders(
                    datasets=[args.dataset],
                    data_path=args.data,
                    atom_types_path=args.atom_types_path,
                    include_near_native=False,
                    normalize_adj=(params_dict['normalize_adj'] == 'True'),
                    normalize_x=(params_dict['normalize_x'] == 'True'),
                    include_contacts=(params_dict['include_contacts'] == 'True'),
                    shuffle=False,
                    number=args.threads,
                    gpu=args.gpu)
            prediction.predict_and_save(
                loaders=loaders,
                network=network,
                prediction_output=current_prediction_output,
                predict_and_save_worker=workers.predict_and_save_worker)
        else:
            utils.output('Predictions of checkpoint {} already computed'.format(checkpoint))

        with open(evaluation_output, 'r') as f:
            saved = f.read()
        if checkpoint in saved:
            utils.output('{} is already evaluated'.format(checkpoint))
            continue

        if not args.predict_only:
            utils.output('Evaluation...')
            results = evaluation.evaluate(current_prediction_output)
            metrics_str = ','.join([str(results[m]) for m in evaluation.ALL_METRICS_NAMES])
            output_str = '{},{}\n'.format(checkpoint, metrics_str)
            with open(evaluation_output, 'a') as f:
                f.write(output_str)
            utils.output('Evaluated checkpoint {}'.format(checkpoint))


if __name__ == '__main__':
    sys.path.append(os.environ['MG_LEARNING_PATH'])
    from src.common import prediction, utils
    from src.sgcn import data, evaluation, networks, workers
    main()
