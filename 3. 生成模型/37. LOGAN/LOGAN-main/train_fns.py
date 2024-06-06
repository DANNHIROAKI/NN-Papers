import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import utils
from utils import toggle_grad, ortho, save_weights
from LOGAN import lat_opt_ngd, lat_opt_gd


def create_train_fn(G, D, GD, z_, y_, ema, state_dict, config):
    def train(x, y):
        G_bs = max(config['G_batch_size'], config['batch_size'])
        G.optim.zero_grad()
        D.optim.zero_grad()
        
        
        
        # How many chunks to split x and y into?
        x = torch.split(x, config['batch_size'])
        y = torch.split(y, config['batch_size'])
        counter = 0

        # Optionally toggle D and G's "require_grad"
        if config['toggle_grads']:
            toggle_grad(D, True)
            toggle_grad(G, False)

        for step_index in range(config['num_D_steps']):
            # If accumulating gradients, loop multiple times before an optimizer step
            D.optim.zero_grad()
            for accumulation_index in range(config['num_D_accumulations']):
                #Use latent optimization
                z_prime = lat_opt_ngd(G, D, z_, G_bs, y_, alpha= 0.9, beta= 5, c_rate=0.5, DOT_reg=False)
                
                D_fake, D_real = GD(z_prime[:config['batch_size']], y_[:config['batch_size']],
                                    x[counter], y[counter], train_G=False,
                                    split_D=config['split_D'])

                # Compute components of D's loss, average them, and divide by
                # the number of gradient accumulations
                D_loss_real, D_loss_fake = loss_hinge_dis(D_fake, D_real)
                D_loss = (D_loss_real + D_loss_fake) / float(config['num_D_accumulations'])
                D_loss.backward()
                counter += 1

            # Optionally apply ortho reg in D
            if config['D_ortho'] > 0.0:
                ortho(D, config['D_ortho'])

            D.optim.step()

        # Optionally toggle "requires_grad"
        if config['toggle_grads']:
            toggle_grad(D, False)
            toggle_grad(G, True)

        # Zero G's gradients by default before training G, for safety
        G.optim.zero_grad()

        # If accumulating gradients, loop multiple times
        for accumulation_index in range(config['num_G_accumulations']):
            
            #Use latent optimization for the generator
            z_prime, euc_norm = lat_opt_ngd(G, D, z_, G_bs, y_, alpha= 0.9, beta= 5, c_rate=0.5, DOT_reg=True)

            
            D_fake = GD(z_prime, y_, train_G=True, split_D=config['split_D'])
            G_loss = loss_hinge_gen(D_fake) / float(config['num_G_accumulations'])
            R_z = euc_norm* config['latent_reg_weight']
            G_loss += R_z
            G_loss = G_loss/ float(config['num_G_accumulations'])
            G_loss.backward()

        # Optionally apply modified ortho reg in G
        if config['G_ortho'] > 0.0:
            # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
            ortho(G, config['G_ortho'], blacklist=[param for param in G.shared.parameters()])
        G.optim.step()

        # If we have an ema, update it, regardless of if we test with it or not
        if config['ema']:
            ema.update(state_dict['itr'])

        out = {'G_loss': float(G_loss.item()),
               'D_loss_real': float(D_loss_real.item()),
               'D_loss_fake': float(D_loss_fake.item())}
        # Return G's loss and the components of D's loss.
        return out

    return train


def loss_hinge_dis(dis_fake, dis_real):
    loss_real = torch.mean(F.relu(1. - dis_real))
    loss_fake = torch.mean(F.relu(1. + dis_fake))
    return loss_real, loss_fake


def loss_hinge_gen(dis_fake):
    loss = -torch.mean(dis_fake)
    return loss


''' This function takes in the model, saves the weights (multiple copies if 
    requested), and prepares sample sheets: one consisting of samples given
    a fixed noise seed (to show how the model evolves throughout training),
    a set of full conditional sample sheets, and a set of interp sheets. '''


def save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y,
                    state_dict, config, experiment_name):
    save_weights(G, D, state_dict, config['weights_root'],
                 experiment_name, None, G_ema if config['ema'] else None)
    # Save an additional copy to mitigate accidental corruption if process
    # is killed during a save (it's happened to me before -.-)
    if config['num_save_copies'] > 0:
        save_weights(G, D, state_dict, config['weights_root'],
                     experiment_name,
                     'copy%d' % state_dict['save_num'],
                     G_ema if config['ema'] else None)
        state_dict['save_num'] = (state_dict['save_num'] + 1) % config['num_save_copies']

    # Use EMA G for samples or non-EMA?
    which_G = G_ema if config['ema'] and config['use_ema'] else G

    # Save a random sample sheet with fixed z and y
    with torch.no_grad():
        if config['parallel']:
            fixed_Gz = nn.parallel.data_parallel(which_G, (fixed_z, which_G.shared(fixed_y)))
        else:
            fixed_Gz = which_G(fixed_z, which_G.shared(fixed_y))
    if not os.path.isdir('%s/%s' % (config['samples_root'], experiment_name)):
        os.mkdir('%s/%s' % (config['samples_root'], experiment_name))
    image_filename = '%s/%s/fixed_samples%d.jpg' % (config['samples_root'],
                                                    experiment_name,
                                                    state_dict['itr'])
    torchvision.utils.save_image(fixed_Gz.float().cpu(), image_filename,
                                 nrow=int(fixed_Gz.shape[0] ** 0.5), normalize=True)
    # For now, every time we save, also save sample sheets
    utils.sample_sheet(which_G,
                       classes_per_sheet=50,
                       num_classes=config['n_classes'],
                       samples_per_class=10, parallel=config['parallel'],
                       samples_root=config['samples_root'],
                       experiment_name=experiment_name,
                       folder_number=state_dict['itr'],
                       z_=z_)
    # Also save interp sheets
    for fix_z, fix_y in zip([False, False, True], [False, True, False]):
        utils.interp_sheet(which_G,
                           num_per_sheet=16,
                           num_midpoints=8,
                           num_classes=config['n_classes'],
                           parallel=config['parallel'],
                           samples_root=config['samples_root'],
                           experiment_name=experiment_name,
                           folder_number=state_dict['itr'],
                           sheet_number=0,
                           fix_z=fix_z, fix_y=fix_y, device='cuda')
