import argparse
import os
import shutil
import sys
import torch
import warnings

from src.common import format, graph, names, utils
from src.sgcn import data, networks, workers


warnings.filterwarnings("ignore")

GRAPH_FOLDER_NAME = 'graph'
PREDICTIONS_FILE_NAME = 'predictions.csv'
TMP_FOLDER_NAME = 'tmp'
LOG_FILE_NAME = 'log'
LOG_HEADER = 'S-GCN: Spherical graph convolutional network for protein structures'

ATOM_TYPES_RELATIVE_PATH = 'metadata/protein_atom_types.txt'
BOND_TYPES_RELATIVE_PATH = 'metadata/bond_types.csv'
ELEMENTS_RADII_RELATIVE_PATH = 'metadata/elements_radii.txt'
VORONOTA_RADII_RELATIVE_PATH = 'metadata/voronota_radii.txt'

VERSIONS_RELATIVE_PATH = 'versions'
DEFAULT_VERSION = 'sgcn_5_casp_8_11'

HELP = '''S-GCN: Spherical graph convolutional network for protein structures
Usage: 
./sgcn -i INPUT -v VORONOTA -M MAPS_GENERATOR [-o OUTPUT] [-m MODEL_VERSION] [-g GPU] [-k] [-V] [-h]

Arguments:
-i, --input          path to the input PDB file or to the directory with PDB files
-v, --voronota       path to Voronota executable file
-M, --maps-generator path to mapsGenerator executable file

Optional arguments:
-o, --output         path to the output directory
-m, --model-version  name of S-GCN version (from sgcn/versions)
-g, --gpu            number of GPU device (if present)
-k, --keep-graph     flag to keep graph data for each model in the output directory
-V, --verbose        flag to print all logs to stdout (including warnings)
-h, --help           flag to print usage help to stdout and exit
'''


class CustomParser(argparse.ArgumentParser):
    def print_help(self):
        print(HELP)

    def error(self, message):
        self.print_help()
        print('Error: {}'.format(message))
        sys.exit(1)


def resource_path(root, relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(root)
    return os.path.join(base_path, relative_path)


def execute(
        input_model_path,
        output_root_path,
        root_path,
        voronota_exec,
        maps_generator_exec,
        params_dict,
        network,
        model_name,
        keep_graph,
        gpu,
        logger):

    logger.info('Started model processing', model_name)
    if keep_graph:
        output_path = utils.path([output_root_path, model_name])
        utils.mkdir(output_path, overwrite=True)
    else:
        output_path = output_root_path
    graph_path = utils.path([output_path, GRAPH_FOLDER_NAME])
    utils.mkdir(graph_path, overwrite=True)
    tmp_path = utils.path([output_path, TMP_FOLDER_NAME])
    utils.mkdir(tmp_path, overwrite=True)
    predictions_path = utils.path([output_path, model_name + '.scores'])

    tmp_model_path = utils.path([tmp_path, 'model'])
    tmp_target_scores_path = utils.path([tmp_path, 'scores'])
    tmp_target_scores_path_expanded = utils.path([tmp_path, 'scores_expanded'])
    tmp_balls_path = utils.path([tmp_path, 'balls'])
    tmp_contacts_path = utils.path([tmp_path, 'contacts'])
    tmp_volumes_path = utils.path([tmp_path, 'volumes'])
    tmp_shelling_path = utils.path([tmp_path, 'shelling'])
    tmp_contacts_path_expanded = utils.path([tmp_path, 'contacts_expanded'])
    tmp_volumes_path_expanded = utils.path([tmp_path, 'volumes_expanded'])
    tmp_shelling_path_expanded = utils.path([tmp_path, 'shelling_expanded'])

    allowed_atom_labels = format.get_atom_labels(resource_path(root_path, ATOM_TYPES_RELATIVE_PATH))
    single_bonds, double_bonds, aromat_bonds = format.get_bonds_types(
        resource_path(root_path, BOND_TYPES_RELATIVE_PATH))

    try:
        graph.build_atom_level_data(
            model_name=model_name,
            input_model_path=input_model_path,
            output_model_path=graph_path,
            allowed_atom_labels=allowed_atom_labels,
            voronota_exec=voronota_exec,
            voronota_radii_path=resource_path(root_path, VORONOTA_RADII_RELATIVE_PATH),
            elements_radii_path=resource_path(root_path, ELEMENTS_RADII_RELATIVE_PATH),
            tmp_model_path=tmp_model_path,
            tmp_target_scores_path=tmp_target_scores_path,
            tmp_target_scores_path_expanded=tmp_target_scores_path_expanded,
            tmp_balls_path=tmp_balls_path,
            tmp_contacts_path=tmp_contacts_path,
            tmp_volumes_path=tmp_volumes_path,
            tmp_shelling_path=tmp_shelling_path,
            tmp_contacts_path_expanded=tmp_contacts_path_expanded,
            tmp_volumes_path_expanded=tmp_volumes_path_expanded,
            tmp_shelling_path_expanded=tmp_shelling_path_expanded,
            single_bonds=single_bonds,
            double_bonds=double_bonds,
            aromat_bonds=aromat_bonds,
            logger=logger,
        )
    except utils.ProcessingException:
        if keep_graph:
            shutil.rmtree(output_path)
        return
    except Exception as eee:
        logger.failure('Got unexpected exception while building atom-level data: {}'.format(eee), model_name)
        if keep_graph:
            shutil.rmtree(output_path)
        return
    logger.info('Built atom-level graph', model_name)

    try:
        graph.build_residue_level_data(
            model_name=model_name,
            model_path=graph_path,
            logger=logger)
    except utils.ProcessingException:
        if keep_graph:
            shutil.rmtree(output_path)
        return
    except Exception as eee:
        if keep_graph:
            shutil.rmtree(output_path)
        logger.failure('Got unexpected exception while building residue-level data: {}'.format(eee), model_name)
        return
    logger.info('Built residue-level graph', model_name)

    try:
        graph.build_spherical_harmonics(
            model_path=graph_path,
            order=int(params_dict['order']),
            maps_generator=maps_generator_exec,
            skip_errors=True)
    except Exception as eee:
        if keep_graph:
            shutil.rmtree(output_path)
        logger.failure('Got unexpected exception while building spherical harmonics: {}'.format(eee), model_name)
        return
    logger.info('Built spherical harmonics', model_name)

    try:
        empty_dataloader = data.DataLoader(
            models_info=None,
            residue_type_to_id=utils.get_residue_type_to_id(resource_path(root_path, ATOM_TYPES_RELATIVE_PATH)),
            shuffle=False,
            include_near_native=False,
            normalize_adj=(params_dict['normalize_adj'] == 'True'),
            normalize_x=(params_dict['normalize_x'] == 'True'),
            include_contacts=(params_dict['include_contacts'] == 'True'),
            description='Empty DataLoader',
            gpu=gpu)
        protein_graph = empty_dataloader.build_graph(graph_path)
        predictions, global_prediction, _, _ = workers.predict_single_graph(protein_graph, network)
    except Exception as ee:
        logger.failure('Got unexpected exception while predicting scores: {}'.format(ee), model_name)
        return
    logger.info('Predicted scores', model_name)

    predictions_df = format.get_pdb_dataframe_legend(utils.path([graph_path, names.MODEL_FILE_NAME]))
    predictions_df['prediction'] = predictions
    predictions_df.to_csv(predictions_path, index=False, sep='\t')
    logger.result('Global prediction: {}'.format(global_prediction), model_name)
    shutil.rmtree(tmp_path)
    if not keep_graph:
        shutil.rmtree(graph_path)


def main(args):
    output_root_path = utils.path([args.output, f'{args.model_version}_output'])
    utils.mkdir(output_root_path)
    log_path = utils.path([output_root_path, LOG_FILE_NAME])
    logger = utils.Logger(log_path, LOG_HEADER, verbose=args.verbose, include_model=os.path.isdir(args.input))
    if args.model_version not in os.listdir(resource_path(args.sgcn_root, VERSIONS_RELATIVE_PATH)):
        logger.error('Version {} does not exist. Please, choose one of the following: {}'.format(
            args.model_version, ', '.join(os.listdir(resource_path(args.sgcn_root, VERSIONS_RELATIVE_PATH)))))
        return
    try:
        state = torch.load(
            resource_path(args.sgcn_root, utils.path([VERSIONS_RELATIVE_PATH, args.model_version])),
            map_location=torch.device('cpu'))
        params_dict = utils.parse_experiment_description(state['model_description'])
        network = networks.get_network(
            name=params_dict['network'],
            features_dim=int(params_dict['features']),
            order=int(params_dict['order']),
            conv_nonlinearity=params_dict['conv_nonlinearity'],
            dropout=float(params_dict.get('dropout', '0.2')))
        network.load_state_dict(state['model_state'])
        if args.gpu:
            network = network.cuda(args.gpu)
        network.eval()
    except Exception as ee:
        logger.failure('Got unexpected exception while preparing network: {}'.format(ee))
        shutil.rmtree(output_root_path)
        return
    logger.info('Loaded S-GCN version "{}"'.format(args.model_version.split('/')[-1]))

    if os.path.isdir(args.input):
        logger.info('Found {} models in directory {}'.format(len(os.listdir(args.input)), args.input))
        for model_file in os.listdir(args.input):
            model_name = model_file.replace('.pdb', '')
            input_model_path = utils.path([args.input, model_file])
            execute(
                input_model_path=input_model_path,
                output_root_path=output_root_path,
                root_path=args.sgcn_root,
                voronota_exec=args.voronota,
                maps_generator_exec=args.maps_generator,
                params_dict=params_dict,
                network=network,
                model_name=model_name,
                keep_graph=args.keep_graph,
                gpu=args.gpu,
                logger=logger)
    else:
        model_name = os.path.basename(args.input).replace('.pdb', '')
        logger.info('Found one model: {}'.format(args.input))
        execute(
            input_model_path=args.input,
            output_root_path=output_root_path,
            root_path=args.sgcn_root,
            voronota_exec=args.voronota,
            maps_generator_exec=args.maps_generator,
            params_dict=params_dict,
            network=network,
            model_name=model_name,
            keep_graph=args.keep_graph,
            gpu=args.gpu,
            logger=logger)


if __name__ == '__main__':
    parser = CustomParser()
    parser.add_argument(
        '-i', '--input',
        action='store',
        type=str,
        required=True,
        help='Path to input model PDB file or to the directory with PDB files')
    parser.add_argument(
        '-v', '--voronota',
        action='store',
        type=str,
        required=True,
        help='Path to Voronota executable file')
    parser.add_argument(
        '-M', '--maps-generator',
        action='store',
        type=str,
        required=True,
        help='Path to mapsGenerator executable file')
    parser.add_argument(
        '-o', '--output',
        action='store',
        type=str,
        required=False,
        default='.',
        help='Path to output directory')
    parser.add_argument(
        '-m', '--model-version',
        action='store',
        type=str,
        required=False,
        default=DEFAULT_VERSION,
        help='Name of S-GCN version')
    parser.add_argument(
        '-g', '--gpu',
        action='store',
        type=int,
        required=False,
        default=None,
        help='Number of GPU device (if present)')
    parser.add_argument(
        '-k', '--keep-graph',
        action='store_true',
        default=False,
        help='Keep graph')
    parser.add_argument(
        '-V', '--verbose',
        action='store_true',
        default=False,
        help='Verbose mode')
    parser.add_argument(
        '-r', '--sgcn-root',
        action='store',
        type=str,
        required=False,
        default='.',
        help='Path to the S-GCN project directory')
    arguments = parser.parse_args()
    try:
        main(arguments)
    except Exception as e:
        print('Fatal error, let us know about it! Exception: {}'.format(e))
