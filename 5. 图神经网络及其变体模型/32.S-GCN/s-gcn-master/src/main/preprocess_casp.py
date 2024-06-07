import argparse
import multiprocessing as mp
import os
import sys


def process(args):
    try:
        graph.preprocess_models_for_casp(**args)
    except Exception as e:
        utils.output('{}: SOMETHING WENT WRONG!!! {}'.format(args['target_name'], e))
        return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', action="store", type=str, required=True)
    parser.add_argument('--targets', action="store", type=str, required=False, default=None)
    parser.add_argument('--output', action="store", type=str, required=True)
    parser.add_argument('--bond_types', action="store", type=str, required=True)
    parser.add_argument('--atom_types', action="store", type=str, required=True)
    parser.add_argument('--elements_radii', action="store", type=str, required=True)
    parser.add_argument('--voronota_radii', action="store", type=str, required=True)
    parser.add_argument('--voronota', action="store", type=str, required=True)
    parser.add_argument('--include_near_native', action="store_true", default=False, required=False)
    parser.add_argument('--nolb_rmsd', action="store", type=float, default=0.9, required=False)
    parser.add_argument('--nolb_samples_num', action="store", type=int, default=50, required=False)
    parser.add_argument('--nolb', action="store", type=str, required=False)
    parser.add_argument('--cadscore', action="store", type=str, required=False, default=None)
    parser.add_argument('--cadscore_window', action="store", type=int, required=False, default=2)
    parser.add_argument('--cadscore_neighbors', action="store", type=int, required=False, default=1)
    parser.add_argument('--sh_featurizer', action="store", type=str, required=False, default=None)
    parser.add_argument('--sh_order', action="store", type=int, required=False, default=None)
    parser.add_argument('--threads', action="store", type=int, default=1, required=False)
    args = parser.parse_args()

    utils.output('Start preprocessing {}'.format(args.models))

    with open(args.atom_types, 'r') as f:
        allowed_atom_labels = set([line.strip() for line in f.readlines()])
    single_bonds, double_bonds, aromat_bonds = format.get_bonds_types(args.bond_types)

    arguments = []
    for target_name in os.listdir(args.models):
        target_path = utils.path([args.targets, target_name + '.pdb']) if args.targets is not None else None
        if target_path is not None and not os.path.exists(target_path):
            target_path = None

        near_native_config = {
            'rmsd': args.nolb_rmsd,
            'samples_num': args.nolb_samples_num,
            'nolb_path': args.nolb
        } if args.include_near_native else None

        arguments.append({
            'target_name': target_name,
            'models_path': args.models,
            'output_path': args.output,
            'allowed_atom_labels': allowed_atom_labels,
            'elements_radii_path': args.elements_radii,
            'voronota_radii_path': args.voronota_radii,
            'single_bonds': single_bonds,
            'double_bonds': double_bonds,
            'aromat_bonds': aromat_bonds,
            'voronota_exec': args.voronota,
            'target_path': target_path,
            'near_native_config': near_native_config,
            'cadscore_exec': args.cadscore,
            'cadscore_window': args.cadscore_window,
            'cadscore_neighbors': args.cadscore_neighbors,
            'sh_order': args.sh_order,
            'maps_generator_exec': args.sh_featurizer,
        })

    if args.threads > 1:
        pool = mp.Pool(processes=args.threads)
        pool.map(process, arguments)
    else:
        for args in arguments:
            process(args)


if __name__ == '__main__':
    sys.path.append(os.environ['MG_LEARNING_PATH'])
    from src.common import format, graph, utils
    main()
