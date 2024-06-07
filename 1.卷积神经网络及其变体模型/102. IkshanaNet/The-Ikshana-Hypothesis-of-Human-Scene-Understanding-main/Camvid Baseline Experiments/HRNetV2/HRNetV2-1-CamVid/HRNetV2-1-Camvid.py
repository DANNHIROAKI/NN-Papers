import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys, time, os, warnings 
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image
from albumentations import (Compose,Resize,Normalize)
import cv2
import matplotlib.pylab as plt
from torch.utils.data import Dataset, DataLoader
import os.path as osp

mean = [0.485, 0.456, 0.406] 
std = [0.229, 0.224, 0.225]
h,w=368,480

transform_train = Compose([ Resize(h,w), 
                Normalize(mean=mean,std=std)])

transform_val = Compose( [ Resize(h,w),
                          Normalize(mean=mean,std=std)])


class CamVidDataSet(Dataset):
    """ 
       CamVidDataSet is employed to load train set
       Args:
        root: the CamVid dataset path, 
        list_path: camvid_train_list.txt, include partial path
    """

    def __init__(self, root='', list_path='', transforms=None):
        self.root = root
        self.list_path = list_path
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        self.transforms = transforms

        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, name.split()[0])
            #print(img_file)
            label_file = osp.join(self.root, name.split()[1])
            #print(label_file)
            self.files.append({
                "img": img_file,
                "label": label_file
            })

        print("length of train set: ", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        #print(image)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        label = cv2.resize(label, (480,368), interpolation=cv2.INTER_NEAREST)
        #print(label)
        if self.transforms is not None:
          augmented= self.transforms(image=np.array(image), label=np.array(label))
          image = augmented['image']
          label = augmented['label'] 
        image = to_tensor(image) 
        label = torch.from_numpy(label).type(torch.long)
        return image , label
train = CamVidDataSet(root='./',list_path='./SegNet/CamVid/train.txt',transforms=transform_train)
val = CamVidDataSet(root='./',list_path='./SegNet/CamVid/val.txt',transforms=transform_val)
test = CamVidDataSet(root='./',list_path='./SegNet/CamVid/test.txt',transforms=transform_val)
#defining Dataloaders
from torch.utils.data import DataLoader
train_dl = DataLoader(train, batch_size=2, shuffle=True)
val_dl = DataLoader(val, batch_size=2, shuffle=False)



import sys
import os
try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

def load_url(url, model_dir='./pretrained', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)

BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.1


__all__ = ['hrnetv2']


model_urls = {
    'hrnetv2': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/hrnetv2_w48-imagenet.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion,
                               momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(num_channels[branch_index] * block.expansion,
                            momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=(height_output, width_output),
                        mode='bilinear',
                        align_corners=False)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HRNetV2(nn.Module):
    def __init__(self, n_class, **kwargs):
        super(HRNetV2, self).__init__()
        extra = {
            'STAGE2': {'NUM_MODULES': 1, 'NUM_BRANCHES': 2, 'BLOCK': 'BASIC', 'NUM_BLOCKS': (4, 4), 'NUM_CHANNELS': (48, 96), 'FUSE_METHOD': 'SUM'},
            'STAGE3': {'NUM_MODULES': 4, 'NUM_BRANCHES': 3, 'BLOCK': 'BASIC', 'NUM_BLOCKS': (4, 4, 4), 'NUM_CHANNELS': (48, 96, 192), 'FUSE_METHOD': 'SUM'},
            'STAGE4': {'NUM_MODULES': 3, 'NUM_BRANCHES': 4, 'BLOCK': 'BASIC', 'NUM_BLOCKS': (4, 4, 4, 4), 'NUM_CHANNELS': (48, 96, 192, 384), 'FUSE_METHOD': 'SUM'},
            'FINAL_CONV_KERNEL': 1
            }

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(Bottleneck, 64, 64, 4)

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)
        last_inp_channels = np.int(np.sum(pre_stage_channels))
        
        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            BatchNorm2d(last_inp_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=n_class,
                kernel_size=1,
                stride=1,
                padding=0) )


    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        BatchNorm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x, return_feature_maps=False):
        size = (x.size()[2:])
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)

        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(
            x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=False)
        x2 = F.interpolate(
            x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=False)
        x3 = F.interpolate(
            x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=False)

        x = torch.cat([x[0], x1, x2, x3], 1)


        x = self.last_layer(x)
        x = torch.nn.functional.interpolate(x, size, mode='bilinear',align_corners=True)
        return x


def hrnetv2(pretrained=False, **kwargs):
    model = HRNetV2(n_class=12, **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls['hrnetv2']), strict=False)

    return model

model =  hrnetv2(pretrained=True)



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model=model.to(device)



criterion = nn.CrossEntropyLoss(reduction="sum")
from torch import optim
opt = optim.SGD(model.parameters(), lr=1e-6, momentum=0.9,nesterov=True)

def loss_batch(loss_func, output, target, opt=None):   
    loss = loss_func(output, target)
    
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item(), None


from torch.optim.lr_scheduler import ReduceLROnPlateau
lr_scheduler = ReduceLROnPlateau(opt, mode='min',factor=0.5, patience=20,verbose=1)

def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

current_lr=get_lr(opt)
print('current lr={}'.format(current_lr))

#loss_history={"train": [],"val": []}

import shutil
def save_ckp(state, is_best, checkpoint_dir, best_model_dir):
    f_path = checkpoint_dir +'checkpoint.pt'
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_dir +'best_model.pt'
        shutil.copyfile(f_path, best_fpath)

checkpoint_dir = "./HRNetV2-1-CamVid/"
best_model_dir = "./HRNetV2-1-CamVid/"

def load_ckp(checkpoint_fpath, model, opt,lr_scheduler):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    opt.load_state_dict(checkpoint['optimizer'])
    lo = (checkpoint['loss'])
    lr_scheduler.load_state_dict(checkpoint['lr'])

    return model, opt, checkpoint['epoch'],lo,lr_scheduler

#ckp_path = "./HRNetV2-1-CamVid/checkpoint.pt"

#model, opt, start_epoch, loss_history, lr_scheduler = load_ckp(ckp_path, model, opt,lr_scheduler)



def loss_epoch(model,loss_func,dataset_dl,sanity_check=False,opt=None):
    running_loss=0.0
    len_data=len(dataset_dl.dataset)
    for xb, yb in dataset_dl:
        xb=xb.to(device)
        yb=yb.to(device)
        output=model(xb)
        loss_b, metric_b = loss_batch(loss_func, output, yb, opt)
        running_loss += loss_b
        if sanity_check is True:
            break
    loss=running_loss/float(len_data)
    return loss, None

import copy
def train_val(model, params):
    num_epochs=params["num_epochs"]
    loss_func=params["loss_func"]
    opt=params["optimizer"]
    train_dl=params["train_dl"]
    val_dl=params["val_dl"]
    sanity_check=params["sanity_check"]
    lr_scheduler=params["lr_scheduler"]
    path2weights=params["path2weights"]

    loss_history={"train": [],"val": []}
    
    best_model_wts = copy.deepcopy(model.state_dict())
    #best_model_wts = copy.deepcopy(torch.load("./HRNetV2-1-CamVid/HRNetV2-1_weights.pt"))

    #best_loss= min(loss_history["val"])
    best_loss=float('inf')
    o = open('./HRNetV2-1-CamVid/HRNetV2-1.txt','a')
    #for epoch in range(start_epoch,num_epochs):
    for epoch in range(num_epochs):
        current_lr=get_lr(opt)

        print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs - 1, current_lr), file=o)

        
        print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs - 1, current_lr))   

        model.train()
        train_loss, _ = loss_epoch(model,loss_func,train_dl,sanity_check,opt)
        loss_history["train"].append(train_loss)
        
        model.eval()
        with torch.no_grad():
            val_loss, _ = loss_epoch(model,loss_func,val_dl,sanity_check)
        loss_history["val"].append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), path2weights)
            print("Copied best model weights!")
            print("Copied best model weights!",file=o)
            is_best = True
        else:
            is_best = False
            
        lr_scheduler.step(val_loss)
        if current_lr != get_lr(opt):
            print("Loading best model weights!")
            print("Loading best model weights!",file=o)
            model.load_state_dict(best_model_wts)

        print("train loss: %.6f" %(train_loss))
        print("train loss: %.6f" %(train_loss),file=o)
        print("val loss: %.6f" %(val_loss))
        print("val loss: %.6f" %(val_loss),file=o)
        print("-"*10)
        print("-"*10,file=o) 
        if epoch %10 == 0:
          print("true")
          checkpoint = {'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': opt.state_dict(),
                        'loss' : loss_history,
                        'lr': lr_scheduler.state_dict()
                        }
          save_ckp(checkpoint, is_best, checkpoint_dir,best_model_dir)


    model.load_state_dict(best_model_wts)
    o.close()
    return model, loss_history



start = time.time()

import os
path2models= "./HRNetV2-1-CamVid/HRNetV2-1_"
if not os.path.exists(path2models):
        os.mkdir(path2models)
params_train={
    "num_epochs": 180,
    "optimizer": opt,
    "loss_func": criterion,
    "train_dl": train_dl,
    "val_dl": val_dl,
    "sanity_check": False,
    "lr_scheduler": lr_scheduler,
    "path2weights": path2models+"weights.pt",}
model,loss_hist=train_val(model,params_train)

end = time.time()
o = open('./HRNetV2-1-CamVid/HRNetV2-1-time.txt','w')

print("TIME TOOK {:3.2f}MIN".format((end - start )/60), file=o)

o.close()

print("TIME TOOK {:3.2f}MIN".format((end - start )/60))


num_epochs=params_train["num_epochs"]
plt.figure(figsize=(30,30))
plt.title("Train-Val Loss")
plt.plot(range(1,num_epochs+1),loss_hist["train"],label="train")
plt.plot(range(1,num_epochs+1),loss_hist["val"],label="val")
plt.ylabel("Loss")
plt.xlabel("Training Epochs")
plt.legend()
plt.savefig('./HRNetV2-1-CamVid/HRNetV2-1.png', dpi = 300)

out_dl =DataLoader(val, batch_size=1, shuffle=False)

t_dl = DataLoader(test, batch_size=1, shuffle=False)

SMOOTH = 1e-6

def iou_pytorch(outputs, labels):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    #outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return thresholded,iou  # Or thresholded.mean() if you are interested in average across the batch


iou_sum = torch.zeros(1)
model.eval()
with torch.no_grad():
    for xb, yb in out_dl:
        yb_pred = model(xb.to(device))
        #yb_pred = yb_pred["out"].cpu()
        yb_pred = yb_pred.cpu()
        #print(yb_pred.shape)
        yb_pred = torch.argmax(yb_pred,axis=1)
        t, i = iou_pytorch(yb_pred,yb)
        iou_sum += i.cpu()

print(iou_sum/101)
o = open('./HRNetV2-1-CamVid/HRNetV2-1_IOU_val.txt','w')

print(iou_sum/101, file=o)

o.close()


i_sum = torch.zeros(1)
with torch.no_grad():
    for xb, yb in t_dl:
        yb_pred = model(xb.to(device))
        #yb_pred = yb_pred["out"].cpu()
        yb_pred = yb_pred.cpu()
        #print(yb_pred.shape)
        yb_pred = torch.argmax(yb_pred,axis=1)
        t, i = iou_pytorch(yb_pred,yb)
        i_sum += i.cpu()

print(i_sum/233)
o = open('./HRNetV2-1-CamVid/HRNetV2-1_IOU_test.txt','w')

print(i_sum/233, file=o)

o.close()

a = loss_hist["train"]
A = [int(x) for x in a]
b = loss_hist["val"]
B = [int(x) for x in b]

import csv

with open('./HRNetV2-1-CamVid/HRNetV2-1-plot.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(A,B))





