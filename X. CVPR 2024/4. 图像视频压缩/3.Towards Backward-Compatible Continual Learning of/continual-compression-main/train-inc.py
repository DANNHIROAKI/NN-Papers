from tqdm import tqdm
from pathlib import Path
import yaml
import argparse
import torch

import cloc

import cloc
from cloc.training import set_logging, set_device, adjust_lr, vr_eval_and_log


def parse_args():
    # ====== set the run settings ======
    parser = argparse.ArgumentParser()
    boa = argparse.BooleanOptionalAction
    # wandb
    parser.add_argument('--wbproject',  type=str,   default='cloc')
    parser.add_argument('--wbgroup',    type=str,   default=None)
    parser.add_argument('--wbmode',     type=str,   default='disabled')
    parser.add_argument('--name',       type=str,   default=None)
    parser.add_argument('--resume',     type=str,   default=None)
    # incremental learning config
    parser.add_argument('--model',      type=str,   default='msh_vr')
    parser.add_argument('--config',     type=str,   default='./configs/data-ca256_0.5.yaml')
    # optimization
    parser.add_argument('--iterations', type=int,   default=100_000)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--lr_sched',   type=str,   default='cosine')
    # pytorch
    parser.add_argument('--compile',    action=boa, default=False)
    parser.add_argument('--workers',    type=int,   default=4)
    cfg = parser.parse_args()

    yaml_path = Path(cfg.config)

    # default settings
    cfg.grad_clip = 2.0
    cfg.wandb_log_interval = 20
    cfg.wbgroup = yaml_path.stem.split('_')[0] if (cfg.wbgroup is None) else cfg.wbgroup

    # merge yaml config to argparse
    with open(yaml_path, mode='r') as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in yaml_dict.items():
        assert not hasattr(cfg, k), f'arg {k} already exists in argparse'
        setattr(cfg, k, v)
    return cfg


def get_main_model(cfg, device):
    model = cloc.get_model(cfg.model, lmb_range=cfg.lmb, pretrained=cfg.pretrained)
    # model.freeze_weights(group=cfg.freeze)
    model.scale_lmb_embedding()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    trainset = cloc.get_dataset(cfg.trainset)
    trainloader = cloc.datasets.make_train_loader(trainset, cfg.batch_size, cfg.workers)
    return model, optimizer, trainloader

def get_replay_model(cfg, device): # for knowledge replay
    replay_model = cloc.get_model(
        cfg.model, lmb_range=cfg.replay_lmb, pretrained=True
    )
    replay_model = replay_model.eval().to(device)
    for p in replay_model.parameters():
        p.requires_grad = False
    replay_set = cloc.get_dataset(cfg.replay_trainset)
    replay_loader = cloc.datasets.make_train_loader(replay_set, cfg.replay_batch_size, cfg.workers)
    return replay_model, replay_loader

def get_valset(cfg):
    valset = cloc.get_dataset(cfg.valset)
    workers = 0 if (cfg.val_batch_size == 1) else cfg.workers//2
    return cloc.datasets.make_val_loader(valset, cfg.val_batch_size, workers=workers)


def replay_step(batch, model, replay_model):
    lmb = replay_model.sample_lmb(n=batch.shape[0])
    with torch.no_grad():
        latents = replay_model.get_latents(batch, lmb)
    loss = model.latents_replay(latents, lmb, batch)
    return loss

def main():
    cfg = parse_args()

    log_dir, wbrun = set_logging(cfg)
    device = set_device()

    model, optimizer, trainloader = get_main_model(cfg, device)
    replay_model, replay_loader = get_replay_model(cfg, device)
    valloader = get_valset(cfg)

    model_cmp = torch.compile(model) if cfg.compile else model

    # ======================== training loops ========================
    pbar = tqdm(range(cfg.iterations), ascii=True)
    for step in pbar:
        if step % cfg.model_val_interval == 0: # evaluation
            vr_eval_and_log(model, valloader, wbrun, step, log_dir, optimizer.state_dict())
            model.train()

        if step % 10 == 0: # learning rate schedule
            adjust_lr(cfg, optimizer, step)

        # training step
        b1 = next(trainloader).to(device=device)
        metrics = model_cmp(b1)

        # knowledge replay
        b2 = next(replay_loader).to(device=device)
        replay_loss = replay_step(b2, model, replay_model)
        metrics['rp-loss'] = replay_loss.item()

        n1, n2 = b1.shape[0], b2.shape[0]
        a1, a2 = n1 / (n1 + n2), n2 / (n1 + n2)
        loss = a1 * metrics['loss'] + a2 * replay_loss
        assert torch.isfinite(loss), f'loss={loss}'

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model_cmp.parameters(), cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad()

        # logging
        msg = ', '.join([f'{k}={float(v):.4f}' for k, v in metrics.items()])
        pbar.set_description(msg)
        if step % cfg.wandb_log_interval == 0: # log to wandb
            log_dict = {f'train/{k}': float(v) for k, v in metrics.items()}
            log_dict['general/grad_norm'] = grad_norm.item()
            for i, pg in enumerate(optimizer.param_groups):
                log_dict[f'general/lr-pg{i}'] = pg['lr']
            wbrun.log(log_dict, step=step)

    # final evaluation
    vr_eval_and_log(model, valloader, wbrun, step, log_dir, optimizer.state_dict())


if __name__ == '__main__':
    main()
