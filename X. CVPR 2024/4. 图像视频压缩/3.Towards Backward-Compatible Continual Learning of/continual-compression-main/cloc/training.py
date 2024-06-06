from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import json
import logging
import torch
import wandb
from timm.utils import random_seed, AverageMeter, unwrap_model

import cloc


def blue_underline(s):
    return f'\u001b[4m\u001b[94m{s}\u001b[0m'


def read_wandb_id(fpath):
    with open(fpath, mode='r') as f:
        s = f.read()
    return s.strip().split('\n')[-1]


def set_logging(cfg):
    print()
    handler = cloc.utils.my_stream_handler()
    logging.basicConfig(handlers=[handler], level=logging.INFO)

    # create folder to save run
    wbgroup = cfg.wbgroup or cfg.valset
    parent_dir = Path(f'runs/{cfg.wbproject}-{wbgroup}')
    if getattr(cfg, 'resume', None) is not None: # resume run
        run_name = cfg.resume
    else: # new run
        run_name = cfg.name or cloc.utils.increment_dir(parent_dir, cfg.model)
    log_dir = parent_dir / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f'Logging run at {blue_underline(log_dir)} \n')

    # get wandb id if exists
    wbid_path = log_dir / 'wandb_id.txt'
    rid = read_wandb_id(wbid_path) if wbid_path.is_file() else None

    # initialize wandb
    wb_name = f'{run_name}: {cfg.model_args}' if getattr(cfg, 'model_args', None) else run_name
    wbrun = wandb.init(
        project=cfg.wbproject, group=wbgroup, name=wb_name, 
        config=cfg, dir='runs/', id=rid, resume='allow', save_code=True, mode=cfg.wbmode
    )
    wbrun.config.log_dir = str(log_dir.resolve())
    with open(wbid_path, mode='a') as f:
        print(wbrun.id, file=f)
    return log_dir, wbrun


def set_device():
    random_seed(2, rank=0)

    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    device = torch.device('cuda:0')
    logging.info(f'Using device {device}: {torch.cuda.get_device_properties(device)}\n')

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    return device


def set_model(cfg):
    kwargs = eval(f'dict({cfg.model_args})')
    model = cloc.get_model(cfg.model, **kwargs)
    logging.info(f'Model name = {cfg.model}, type = {type(model)}, args = {kwargs}')
    cfg.trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'Number of trainable parameters: {cfg.trainable_params/1e6:.2f}M \n')
    return model


def adjust_lr(cfg, optimizer, t):
    lrf = cloc.utils.get_lr_factor(cfg.lr_sched, t, cfg.iterations, T_warmup=0, lrf_min=0.1)
    for param_group in optimizer.param_groups:
        param_group['lr'] = cfg.lr * lrf
    return lrf


@torch.inference_mode()
def var_rate_evaluate(model: torch.nn.Module, testloader):
    device = next(model.parameters()).device
    assert hasattr(model, 'default_eval_lmbs') and hasattr(model, 'default_lmb')

    all_lmb_metrics = defaultdict(list)
    for lmb in tqdm(model.default_eval_lmbs, ascii=True, position=0, leave=True):
        model.default_lmb = float(lmb)

        one_lmb_met = defaultdict(AverageMeter)
        for batch in testloader: # enumerate all images
            metrics = model.forward(batch.to(device=device)) # dict[str, float]
            for k, v in metrics.items():
                one_lmb_met[k].update(float(v), n=batch.shape[0])

        # save results
        all_lmb_metrics['lmb'].append(lmb)
        for k, meter in one_lmb_met.items():
            all_lmb_metrics[k].append(meter.avg)
    return all_lmb_metrics # dict[str, list]

def print_dict_of_list(dict_of_list):
    for k, vlist in dict_of_list.items():
        assert isinstance(vlist, list)
        vlist_str = '[' + ', '.join([f'{v:.12f}'[:6] for v in vlist]) + ']'
        logging.info(f"'{k:<6s}': {vlist_str}")

def _vr_metrics_to_wandb(results, prefix=''):
    # results: dict of list
    print_dict_of_list(results)

    wandb_data = dict()
    for i, lmb in enumerate(results['lmb']):
        group = f'{prefix}val-lmb{round(lmb)}'
        wandb_data.update({
            f'{group}/loss': results['loss'][i],
            f'{group}/bpp':  results['bpp'][i],
            f'{group}/psnr': results['psnr'][i],
        })
    return wandb_data

def _read_json(fpath):
    with open(fpath, mode='r') as f:
        stats = json.load(fp=f)
    stats = stats.get('results', stats)
    return stats

@torch.inference_mode()
def vr_eval_and_log(model, dataloader, wbrun, step, log_dir, opt_state_dict=None):
    # Evaluation
    metrics = var_rate_evaluate(model.eval(), dataloader) # dict[str, list]

    # wandb log
    log_dict = {'general/iter': step}
    log_dict.update(_vr_metrics_to_wandb(metrics))
    # bd-rate
    anchor = _read_json('results/kodak-vtm22.0.json')
    log_dict['val-bd-rate/kodak-vtm22.0'] = cloc.utils.bd_rate(
        anchor['bpp'], anchor['psnr'], metrics['bpp'], metrics['psnr']
    )
    wbrun.log(log_dict, step=step)

    # save last checkpoint
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': opt_state_dict,
        'iter': step,
        'metrics': metrics,
    }
    ckpt_path = log_dir / 'last.pt'
    torch.save(checkpoint, ckpt_path)
    logging.info(f'Last checkpoint saved to {ckpt_path} \n')


@torch.inference_mode()
def vr_evaluate(model, dataloader, wbrun, step):
    # Evaluation
    model = unwrap_model(model).eval()
    metrics = var_rate_evaluate(model, dataloader) # dict[str, list]

    # wandb log
    log_dict = {'general/iter': step}
    log_dict.update(_vr_metrics_to_wandb(metrics))
    # bd-rate
    anchor = _read_json('results/kodak-vtm22.0.json')
    log_dict['val-bd-rate/kodak-vtm22.0'] = cloc.utils.bd_rate(
        anchor['bpp'], anchor['psnr'], metrics['bpp'], metrics['psnr']
    )
    wbrun.log(log_dict, step=step)
    return metrics


def save_checkpoints(log_dir, step, model, model_ema=None, optimizer=None):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': None if (optimizer is None) else optimizer.state_dict(),
        'step': step
    }
    torch.save(checkpoint, log_dir / 'last.pt')
    if model_ema is not None:
        checkpoint = {
            'model': unwrap_model(model_ema).state_dict(),
            'step': step
        }
        torch.save(checkpoint, log_dir / 'last_ema.pt')
    logging.info(f'Lastest checkpoints saved to {log_dir} \n')
