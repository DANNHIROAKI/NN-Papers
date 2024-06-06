"""train WGAN"""
import os
import random
import json
import time
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.dataset as ds
import mindspore.ops as ops
from mindspore.common import initializer as init
import mindspore.common.dtype as mstype
from mindspore import context
from mindspore.context import ParallelMode
from mindspore import communication
from mindspore.communication.management import get_group_size, get_rank
from mindspore.train.serialization import load_checkpoint, load_param_into_net, save_checkpoint
from PIL import Image
import numpy as np

from src.dataset import create_dataset
from src.dcgan_model import DcganG, DcganD
from src.dcgannobn_model import DcgannobnG
from src.cell import GenWithLossCell, DisWithLossCell, GenTrainOneStepCell, DisTrainOneStepCell
from src.args import get_args

if __name__ == '__main__':
    t_begin = time.time()
    args_opt = get_args('train')
    print(args_opt)

    # init context
    target = args_opt.device_target
    context.set_context(mode=context.GRAPH_MODE, device_target=target)

    if args_opt.run_distribute:
        communication.management.init()
        device_num = get_group_size()
        print("group_size(device_num) is: ", device_num)
        rank_id = get_rank()
        print("rank_id is: ", rank_id)
        # set auto parallel context
        context.set_auto_parallel_context(device_num=device_num,
                                          gradients_mean=True,
                                          parallel_mode=ParallelMode.DATA_PARALLEL)
        if args_opt.device_target == 'Ascend':
            context.set_context(device_id=args_opt.device_id)
    else:
        device_num = 1
        rank_id = 0
        context.set_context(device_id=args_opt.device_id)

    # whether train on modelarts or local server
    if not args_opt.is_modelarts:
        if args_opt.experiment is None:
            args_opt.experiment = 'samples'
        os.system('mkdir {0}'.format(args_opt.experiment))

        dataset = create_dataset(args_opt.dataroot, args_opt.dataset, args_opt.batchSize, args_opt.imageSize, 1,
                                 args_opt.workers, target)

    else:
        import moxing as mox
        if args_opt.experiment is None:
            args_opt.experiment = '/cache/train_output'
        os.system('mkdir {0}'.format(args_opt.experiment))
        data_name = 'LSUN-bedroom.zip'
        local_data_url = '/cache/data_path/'
        mox.file.copy_parallel(src_url=args_opt.data_url, dst_url=local_data_url)
        zip_command = "unzip -o -q %s -d %s" % (local_data_url + data_name, local_data_url)
        os.system(zip_command)
        print("Unzip success!")

        dataset = create_dataset(local_data_url, args_opt.dataset, args_opt.batchSize, args_opt.imageSize, 1,
                                 args_opt.workers, target)

    data_loader = dataset.create_dict_iterator()
    length = dataset.get_dataset_size()
    # fix seed
    print("Random Seed: ", args_opt.manualSeed)
    random.seed(args_opt.manualSeed)
    ds.config.set_seed(args_opt.manualSeed)
    mindspore.common.set_seed(args_opt.manualSeed)

    # initialize hyperparameters
    nz = int(args_opt.nz)
    ngf = int(args_opt.ngf)
    ndf = int(args_opt.ndf)
    nc = int(args_opt.nc)
    n_extra_layers = int(args_opt.n_extra_layers)

    # write out generator config to generate images together wth training checkpoints
    generator_config = {"imageSize": args_opt.imageSize, "nz": nz, "nc": nc, "ngf": ngf,
                        "n_extra_layers": n_extra_layers, "noBN": args_opt.noBN}

    with open(os.path.join(args_opt.experiment, "generator_config.json"), 'w') as gcfg:
        gcfg.write(json.dumps(generator_config) + "\n")

    def init_weight(net):
        """initial net weight"""
        for _, cell in net.cells_and_names():
            if isinstance(cell, (nn.Conv2d, nn.Conv2dTranspose)):
                cell.weight.set_data(init.initializer(init.Normal(0.02), cell.weight.shape))
            elif isinstance(cell, nn.BatchNorm2d):
                cell.gamma.set_data(init.initializer(Tensor(np.random.normal(1, 0.02, cell.gamma.shape), \
                mstype.float32), cell.gamma.shape))
                cell.beta.set_data(init.initializer('zeros', cell.beta.shape))


    def save_image(img, img_path):
        """save image"""
        mul = ops.Mul()
        add = ops.Add()
        if isinstance(img, Tensor):
            img = mul(img, 255 * 0.5)
            img = add(img, 255 * 0.5)

            img = img.asnumpy().astype(np.uint8).transpose((0, 2, 3, 1))

        elif not isinstance(img, np.ndarray):
            raise ValueError("img should be Tensor or numpy array, but get {}".format(type(img)))

        IMAGE_SIZE = 64  # Image size
        IMAGE_ROW = 8  # Row num
        IMAGE_COLUMN = 8  # Column num
        PADDING = 2  # Interval of small pictures
        to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE + PADDING * (IMAGE_COLUMN + 1),
                                     IMAGE_ROW * IMAGE_SIZE + PADDING * (IMAGE_ROW + 1)))  # create a new picture
        # cycle
        ii = 0
        for y in range(1, IMAGE_ROW + 1):
            for x in range(1, IMAGE_COLUMN + 1):
                from_image = Image.fromarray(img[ii])
                to_image.paste(from_image, ((x - 1) * IMAGE_SIZE + PADDING * x, (y - 1) * IMAGE_SIZE + PADDING * y))
                ii = ii + 1

        to_image.save(img_path)  # save


    # define net----------------------------------------------------------------------------------------------
    # Generator
    if args_opt.noBN:
        netG = DcgannobnG(args_opt.imageSize, nz, nc, ngf, n_extra_layers)
    else:
        netG = DcganG(args_opt.imageSize, nz, nc, ngf, n_extra_layers)

    # write out generator config to generate images together wth training checkpoints
    generator_config = {"imageSize": args_opt.imageSize, "nz": nz, "nc": nc, "ngf": ngf,
                        "n_extra_layers": n_extra_layers, "noBN": args_opt.noBN}
    with open(os.path.join(args_opt.experiment, "generator_config.json"), 'w') as gcfg:
        gcfg.write(json.dumps(generator_config) + "\n")

    init_weight(netG)

    if args_opt.netG != '':  # load checkpoint if needed
        load_param_into_net(netG, load_checkpoint(args_opt.netG))
    print(netG)

    netD = DcganD(args_opt.imageSize, nz, nc, ndf, n_extra_layers)
    init_weight(netD)

    if args_opt.netD != '':
        load_param_into_net(netD, load_checkpoint(args_opt.netD))
    print(netD)

    input1 = Tensor(np.zeros([args_opt.batchSize, 3, args_opt.imageSize, args_opt.imageSize]), dtype=mstype.float32)
    noise = Tensor(np.zeros([args_opt.batchSize, nz, 1, 1]), dtype=mstype.float32)
    fixed_noise = Tensor(np.random.normal(0, 1, size=[args_opt.batchSize, nz, 1, 1]), dtype=mstype.float32)

    # setup optimizer
    if args_opt.adam:
        optimizerD = nn.Adam(netD.trainable_params(), learning_rate=args_opt.lrD, beta1=args_opt.beta1, beta2=.999)
        optimizerG = nn.Adam(netG.trainable_params(), learning_rate=args_opt.lrG, beta1=args_opt.beta1, beta2=.999)
    else:
        optimizerD = nn.RMSProp(netD.trainable_params(), learning_rate=args_opt.lrD, decay=0.99)
        optimizerG = nn.RMSProp(netG.trainable_params(), learning_rate=args_opt.lrG, decay=0.99)

    G_with_loss = GenWithLossCell(netG, netD)
    D_with_loss = DisWithLossCell(netG, netD)

    netG_train = GenTrainOneStepCell(G_with_loss, optimizerG)
    netD_train = DisTrainOneStepCell(D_with_loss, optimizerD, args_opt.clamp_lower, args_opt.clamp_upper)

    netG_train.set_train()
    netD_train.set_train()

    global_steps = 0

    t0 = time.time()
    # Train
    for epoch in range(args_opt.niter):  # niter: the num of epoch
        for i, data in enumerate(data_loader):
            real = data['image']
            noise = Tensor(np.random.normal(0, 1, size=[args_opt.batchSize, nz, 1, 1]), dtype=mstype.float32)

            # Update D network
            loss_D = netD_train(real, noise)
            # Update G network
            fake_img, loss_G = netG_train(noise)

            t1 = time.time()
            global_steps += 1
            if global_steps % 5000 == 0 and rank_id == 0:
                print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f'
                      % (epoch, args_opt.niter, i, length, global_steps,
                         loss_D.asnumpy(), loss_G.asnumpy()))
                print('step_cost: %.4f seconds' % (float(t1 - t0)))
                fake = netG(fixed_noise)
                save_image(real, '{0}/real_samples_{1}.png'.format(args_opt.experiment, epoch))
                save_image(fake, '{0}/fake_samples_{1}_{2}.png'.format(args_opt.experiment, epoch, global_steps))

            t0 = t1
        if rank_id == 0:
            save_checkpoint(netD, '{0}/netD_epoch_{1}.ckpt'.format(args_opt.experiment, epoch))
            save_checkpoint(netG, '{0}/netG_epoch_{1}.ckpt'.format(args_opt.experiment, epoch))

    if args_opt.is_modelarts and rank_id == 0:
        mox.file.copy_parallel(src_url='/cache/train_output', dst_url=args_opt.train_url)

    t_end = time.time()
    print('total_cost: %.4f seconds' % (float(t_end - t_begin)))

    print("Train success!")
