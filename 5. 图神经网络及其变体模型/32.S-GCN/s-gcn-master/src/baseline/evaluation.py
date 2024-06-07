import numpy as np
import os
import pandas as pd

from scipy.stats import pearsonr, rankdata, spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from src.common import utils


RANKING_METRICS_NAMES = ['z_score', 'rank']
OTHER_METRICS_NAMES = ['mse', 'r2', 'pearson', 'spearman']
SCORES_TYPES = ['local', 'global']
APPROACHES = ['decoy', 'global']
ALL_METRICS_NAMES = RANKING_METRICS_NAMES + [
    '{}_{}_{}'.format(m, st, a)
    for m in OTHER_METRICS_NAMES
    for st in SCORES_TYPES
    for a in APPROACHES
]
METRIC_TO_FUNCTION = {
    'mse': mean_squared_error,
    'r2': r2_score,
    'pearson': lambda x, y: pearsonr(x, y)[0],
    'spearman': lambda x, y: spearmanr(x, y)[0],
}
AVG_FUNCTIONS = {
    'z_score': np.mean,
    'rank': np.mean,
    'mse': np.mean,
    'r2': np.mean,
    'pearson': utils.fisher_mean,
    'spearman': utils.fisher_mean
}


def evaluate_target(target_path):
    local_y_gt = []
    local_y_pred = []
    global_y_gt = []
    global_y_pred = []
    models = [fname.split('.')[0] for fname in os.listdir(target_path) if fname.endswith('.csv')]
    for model in models:
        model_df = pd.read_csv(utils.path([target_path, model + '.csv']))
        model_local_y_gt = model_df.y.tolist()
        model_local_y_pred = model_df.pred.to_list()
        model_global_y_gt = float(open(utils.path([target_path, model + '.y'])).read().strip())
        model_global_y_pred = np.mean(model_local_y_pred)
        local_y_gt += model_local_y_gt
        local_y_pred += model_local_y_pred
        global_y_gt.append(model_global_y_gt)
        global_y_pred.append(model_global_y_pred)
    global_y_gt = global_y_gt
    global_y_pred = global_y_pred
    z_scores = dict(zip(models, utils.calculate_z_scores(global_y_gt)))
    ranks = dict(zip(models, rankdata(-np.array(global_y_gt))))
    choice = models[int(np.argmax(global_y_pred))]

    scores = {
        'local_y_gt': local_y_gt,
        'local_y_pred': local_y_pred,
        'global_y_gt': global_y_gt,
        'global_y_pred': global_y_pred,
    }
    ranking_metrics = {
        'z_score': z_scores[choice],
        'rank': ranks[choice]
    }
    local_decoy_metrics = {
        m: f(local_y_gt, local_y_pred)
        for m, f in METRIC_TO_FUNCTION.items()
    }
    global_decoy_metrics = {
        m: f(global_y_gt, global_y_pred)
        for m, f in METRIC_TO_FUNCTION.items()
    }
    return scores, ranking_metrics, local_decoy_metrics, global_decoy_metrics


def evaluate(predictions_path):
    scores = {
        'local_y_gt': [],
        'local_y_pred': [],
        'global_y_gt': [],
        'global_y_pred': [],
    }
    ranking_metrics = {
        'z_score': [],
        'rank': []
    }
    local_decoy_metrics = {
        m: []
        for m in METRIC_TO_FUNCTION.keys()
    }
    global_decoy_metrics = {
        m: []
        for m in METRIC_TO_FUNCTION.keys()
    }
    for target_name in os.listdir(predictions_path):
        if 'T' in target_name:
            scrs, rm, ldm, gdm = evaluate_target(utils.path([predictions_path, target_name]))
            for s in scrs.keys():
                scores[s] += scrs[s]
            for m in rm.keys():
                ranking_metrics[m].append(rm[m])
            for m in METRIC_TO_FUNCTION.keys():
                local_decoy_metrics[m].append(ldm[m])
                global_decoy_metrics[m].append(gdm[m])
    ranking_metrics_avg = {
        mk: AVG_FUNCTIONS[mk](mv)
        for mk, mv in ranking_metrics.items()
    }
    local_decoy_metrics_avg = {
        '{}_local_decoy'.format(mk): AVG_FUNCTIONS[mk](mv)
        for mk, mv in local_decoy_metrics.items()
    }
    global_decoy_metrics_avg = {
        '{}_global_decoy'.format(mk): AVG_FUNCTIONS[mk](mv)
        for mk, mv in global_decoy_metrics.items()
    }
    local_global_metrics = {
        '{}_local_global'.format(m): f(scores['local_y_gt'], scores['local_y_pred'])
        for m, f in METRIC_TO_FUNCTION.items()
    }
    global_global_metrics = {
        '{}_global_global'.format(m): f(scores['global_y_gt'], scores['global_y_pred'])
        for m, f in METRIC_TO_FUNCTION.items()
    }
    result = {}
    result.update(ranking_metrics_avg)
    result.update(local_decoy_metrics_avg)
    result.update(global_decoy_metrics_avg)
    result.update(local_global_metrics)
    result.update(global_global_metrics)
    return result
