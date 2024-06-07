
import argparse
import os
import numpy as np
import torch
from time import perf_counter as timer
from datetime import datetime

import pinvgcn
import pinvgcn.hypergraphs

### PARSE ARGUMENTS

parser = argparse.ArgumentParser(description='Perform several runs of training and testing a Pseudoinverse GCN on a graph dataset.')

parser.add_argument('dataset', help='Name of the dataset to be loaded')
parser.add_argument('coefficients', nargs='?', default='independent-parts', 
    help='Name of the coefficient setup. Default: independent-parts')

parser.add_argument('-a', '--alpha', type=float, default=1, help='Filter shape parameter. Default: 1')
parser.add_argument('-b', '--beta', type=float, default=1, help='Filter shape parameter. Default: 1')
parser.add_argument('-c', '--gamma', type=float, default=1, help='Filter shape parameter. Default: 1')

parser.add_argument('-n', '--num-runs', default=1, type=int, metavar='N',
    help='Number of runs to be performed')
parser.add_argument('-s', '--split-size', type=int, default=None, metavar='S',
    help="Randomly split the nodes into training and test set using S training samples per class")
parser.add_argument('-f', '--fixed-splits', action='store_true', default=False,
    help="Use the fixed splits given with the data")
parser.add_argument('-R', '--categorical-regularization', type=float, default=0.0, nargs='?', const=1.0, metavar='FACTOR',
    help="Add new hyperedges based on the categorical node features")
parser.add_argument('-r', '--rank', type=int, default=None, metavar='R',
    help='Perform a low-rank approximation with this target rank')
parser.add_argument('-l', '--loops', type=float, default=0.0, metavar='WEIGHT',
    help='Add self loop edges with the given weight. If loops are already present, their weight is increased')
parser.add_argument('--partial-eigs', action='store_true', default=False,
    help='Only compute the required number of eigenvalues instead of a full decomposition')
parser.add_argument('--hidden', nargs='*', type=int, default=[32], metavar='H', 
    help='Hidden layer widths')
parser.add_argument('--dropout', type=float, default=0.5, metavar='RATE',
    help='Dropout rate')
parser.add_argument('--learning-rate', default=0.05, type=float, metavar='LR',
    help='Learning rate')
parser.add_argument('--weight-decay', default=5e-4, type=float, metavar='DECAY',
    help='Weight decay on weight matrices (not on bias vectors)')
parser.add_argument('--epochs', default=500, type=int, metavar='N',
    help='Number of training epochs per run')
parser.add_argument('--no-bias', default=False, action='store_true',
    help='Disable bias')
parser.add_argument('--disable-cuda', action='store_true', default=False,
    help='Disable CUDA acceleration')
parser.add_argument('--no-save', action='store_true', default=False,
    help='Disable saving results')
parser.add_argument('--no-fixed-seeds', action='store_true', default=False,
    help='Disable fixed seeds. Also implies that the results will not be saved')
parser.add_argument('--track-weights', action='store_true', default=False,
    help='Keep track of the average entries in the weight matrices for each basis filter function')
parser.add_argument('--silent-runs', action='store_true', default=False,
    help='Don''t print a line after each run')
parser.add_argument('--print-filter-values', action='store_true', default=False,
    help='Print the values of each filter basis function, evaluated in the computed eigenvalues')
parser.add_argument('--repeat-setup', action='store_true', default=False,
    help='In each run, repeat the setup process')


args = parser.parse_args()

start_time = datetime.now()


### PREPARATIONS

if args.no_fixed_seeds and not args.no_save:
    print('no-fixed-seeds option given. Results will not be reproducible and hence not be saved.')
    args.no_save = True

base_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(base_dir, 'data')

if not args.disable_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


### SETUP

data = pinvgcn.hypergraphs.load_hypergraph_data(args.dataset, data_dir, 
                                                categorical_regularization=args.categorical_regularization)

# print("Number of nodes: {}".format(data.x.shape[0]))
# print("Number of hyperedges: {}".format(data.hyperedge_index[0].max().item()+1))
# print("Incidence nonzeros: {}".format(data.hyperedge_index.shape[1]))
# print("Average hyperedge weight: {}".format(data.hyperedge_weight.mean().item()))

tic = timer()

setup_transform = pinvgcn.hypergraphs.HypergraphSpectralSetup(rank=args.rank, 
                                                              partial_eigs=args.partial_eigs,
                                                              eig_tol=1e-3)
if args.repeat_setup:
    orig_data = data
else:
    data = setup_transform(data)
    print("Computed {} nonzero eigenvalues between {:.4f} and {:.4f}".format(
        data.nonzero_w.shape[0], data.nonzero_w.min().item(), data.nonzero_w.max().item()))
    data = data.to(device)

coeffs = pinvgcn.get_coefficient_preset(args.coefficients, alpha=args.alpha, beta=args.beta, gamma=args.gamma)

if args.print_filter_values:
    pinvgcn.print_filter_values(coeffs, data)

model = pinvgcn.PinvGCN(coeffs, data.num_features, data.num_classes, hidden_channels=args.hidden, 
                        dropout=args.dropout, bias=not args.no_bias)
model.to(device)

optimizer = torch.optim.Adam(model.split_parameters(args.weight_decay), lr=args.learning_rate)

setup_time = timer() - tic
print('Setup done in {:.4} seconds'.format(setup_time))


### EXPERIMENT RUNS

if args.repeat_setup:
    setup_times = []
training_times = []
accuracies = []
avg_weights = 0

try:
    for run in range(args.num_runs):
        if not args.no_fixed_seeds:
            pinvgcn.set_seed(run)
        
        if args.split_size:
            pinvgcn.random_split(data, args.split_size)
        elif args.fixed_splits:
            pinvgcn.fixed_split(data, run)
    
        if args.repeat_setup:
            tic = timer()
            
            data = setup_transform(orig_data)
            data = data.to(device)
            
            t0 = timer() - tic
            setup_times.append(setup_time + t0)
            
            
        tic = timer()
        
        model.reset_parameters()
        
        input = model.preconvolve_input(data, data.x)
        
        model.run_training(data, input, optimizer, num_epochs=args.epochs)
        
        t = timer() - tic
        training_times.append(t)
    
        acc = model.eval_accuracy(data, input)
        accuracies.append(acc)
        
        if args.track_weights:
            avg_run_weights = model.average_absolute_weight_entries()
            avg_weights += avg_run_weights
        
        if not args.silent_runs:
            s = 'Run {: 4d}/{}: '.format(run+1, args.num_runs)
            if args.repeat_setup:
                s += 'Additional setup time {:.4f} s, '.format(t0)
            s += 'Training time {:.4f} s, accuracy {:8.4f} %'.format(t, 100*acc)
            if args.track_weights:
                with np.printoptions(precision=3, suppress=True):
                    s += ', avg. abs. weights: ' + ', '.join('L{} {}'.format(i+1, ww) for i, ww in enumerate(avg_run_weights))
            print(s)
    
except KeyboardInterrupt:
    print("Experiment stopped early due to KeyboardInterrupt.")
    if len(accuracies) == 0:
        raise
    status = 'KeyboardInterrupt'
else:
    status = 'Finished'
    
    
print('###')
pinvgcn.print_results(accuracies, setup_times if args.repeat_setup else setup_time, training_times)

if args.track_weights:
    avg_weights /= len(accuracies)
    with np.printoptions(precision=3, suppress=True):
        weights_str = ', '.join('Layer {} {}'.format(i+1, ww) for i, ww in enumerate(avg_weights)) \
                    + ', Combined {}'.format(avg_weights.mean(axis=0))
    print(' - Average absolute weight entries:', weights_str)
    
print('###')



### SAVE RESULTS

if not args.no_save:
    results_dir = os.path.join(base_dir, 'results', 'hypergraphs')
    
    dataset_name = data.name if 'name' in data else args.dataset
    
    if args.categorical_regularization > 0:
        dataset_name += '_catreg{:g}'.format(args.categorical_regularization)
    
    if args.split_size is not None:
        dataset_name += '_split{}'.format(args.split_size)
    elif args.fixed_splits is not None:
        dataset_name += '_fixedsplits'
    
    architecture_name = 'PinvGCN_' + args.coefficients
    for p in ['alpha','beta','gamma']:
        val = getattr(args, p)
        if val != 1:
            architecture_name += '_{}{}'.format(p,val)
    if args.rank is not None:
        architecture_name += '_rank{}'.format(args.rank)
    if args.no_bias:
        architecture_name += '_nobias'
    
    pinvgcn.save_results(
        results_dir, dataset_name, architecture_name,
        accuracies, setup_times if args.repeat_setup else setup_time, training_times,
        args.__dict__, 
        {'Avg. abs. weights': weights_str if args.track_weights else 'Not tracked',
         'Start time': start_time.strftime("%b %d, %Y, %H:%M:%S"),
         'End time': datetime.now().strftime("%b %d, %Y, %H:%M:%S")
        },
        status = status, file = __file__)

