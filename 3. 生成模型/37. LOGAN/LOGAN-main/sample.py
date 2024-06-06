import os
import torch
import torchvision
from LOGAN import *
import utils
import shutil


def trunc_trick(bs, z_dim, bound=0.8):
    z = torch.randn(bs, z_dim)
    while z.abs().max() > bound:
        z = torch.where(z.abs() <= bound, z, torch.randn_like(z))
    return z


def collect_bn_stats(G, n_samples, config, device):
    im_batch_size = config['n_classes']
    G.train()

    for i_batch in range(0, n_samples, im_batch_size):
        with torch.no_grad():
            z = torch.randn(im_batch_size, G.dim_z, device=device)
            y = torch.arange(im_batch_size).to(device)
            _images = G(z, G.shared(y)).float().cpu()


def generate_images(out_dir, G, n_images, config, device):
    im_batch_size = config['n_classes']
    z_bound = config['trunc_z']
    if z_bound > 0.0:
        print(f'Truncating z to (-{z_bound}, {z_bound})')

    for i_batch in range(0, n_images, im_batch_size):
        with torch.no_grad():
            if z_bound > 0.0:
                z = trunc_trick(im_batch_size, G.dim_z, bound=z_bound).to(device)
            else:
                z = torch.randn(im_batch_size, G.dim_z, device=device)
            y = torch.arange(im_batch_size).to(device)
            images = G(z, G.shared(y)).float().cpu()

        if i_batch + im_batch_size > n_images:
            n_last_images = n_images - i_batch
            print(f'Taking only {n_last_images} images from the last batch...')
            images = images[:n_last_images]

        for i_image, image in enumerate(images):
            fname = os.path.join(out_dir, f'image_{i_batch+i_image:05d}.png')
            image = utils.denorm(image)
            torchvision.utils.save_image(image, fname)


def run(config):
    # Prepare state dict, which holds things like epoch # and itr #
    state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'config': config}

    # update config (see train.py for explanation)
    config['resolution'] = 256
    config['n_classes'] = 40
    config['G_activation'] = utils.activation_dict[config['G_nl']] #leaky relu for LOGAN
    config['D_activation'] = utils.activation_dict[config['D_nl']]
    config = utils.update_config_roots(config)
    config['skip_init'] = True
    config['no_optim'] = True
    device = 'cuda'

    # Seed RNG
    utils.seed_rng(config['seed'])
    # Setup cudnn.benchmark for free speed
    torch.backends.cudnn.benchmark = True

    experiment_name = (config['experiment_name'] if config['experiment_name']
                       else 'PXDgen')
    print('Experiment name is %s' % experiment_name)

    G = Generator(**config).cuda()

    # Load weights
    print('Loading weights...')
    # Here is where we deal with the ema--load ema weights or load normal weights
    utils.load_weights(G if not (config['use_ema']) else None, None, state_dict,
                       config['weights_root'], experiment_name, config['load_weights'],
                       G if config['ema'] and config['use_ema'] else None,
                       strict=False, load_optim=False)

    if config['use_ema']:
        collect_bn_stats(G, 500, config, device)
    if config['G_eval_mode']:
        print('Putting G in eval mode..')
        G.eval()
    else:
        print('G is in %s mode...' % ('training' if G.training else 'eval'))

    out_dir = config['samples_root']
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    print('Generating images..')
    generate_images(out_dir, G, config['sample_num'], config, device)
    shutil.make_archive('images', 'zip', out_dir)


def main():
    # parse command line and run
    parser = utils.prepare_parser()
    parser = utils.add_sample_parser(parser)
    config = vars(parser.parse_args())
    print(config)
    run(config)


if __name__ == '__main__':
    main()
