import torch.multiprocessing as mp

from src.common import names, utils


def predict_and_save(loaders, network, prediction_output, predict_and_save_worker):
    for loader in loaders:
        for model_info in loader.models_info:
            target_name = model_info[names.TARGET_NAME_FIELD]
            utils.mkdir(utils.path([prediction_output, target_name]))

    network.eval()
    processes = []
    for i, loader in enumerate(loaders):
        kwargs = {
            'network': network,
            'loader': loader,
            'output_path': prediction_output,
            'p_id': i
        }
        p = mp.Process(target=predict_and_save_worker, kwargs=kwargs)
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
