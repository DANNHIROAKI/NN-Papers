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


train_ds = CamVidDataSet(root='./',list_path='./SegNet/CamVid/train.txt',transforms=transform_train)
val = CamVidDataSet(root='./',list_path='./SegNet/CamVid/val.txt',transforms=transform_val)
test = CamVidDataSet(root='./',list_path='./SegNet/CamVid/test.txt',transforms=transform_val)
train , dump = torch.utils.data.random_split(train_ds, [91,276], generator=torch.Generator().manual_seed(42))
print(len(train))
print(len(dump))


#defining Dataloaders
from torch.utils.data import DataLoader
train_dl = DataLoader(train, batch_size=2, shuffle=True)
val_dl = DataLoader(val, batch_size=2, shuffle=False)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


model = UNet(n_channels=3,n_classes=12,bilinear=False)

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

checkpoint_dir = "./UNet-3-CamVid/"
best_model_dir = "./UNet-3-CamVid/"

def load_ckp(checkpoint_fpath, model, opt,lr_scheduler):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    opt.load_state_dict(checkpoint['optimizer'])
    lo = (checkpoint['loss'])
    lr_scheduler.load_state_dict(checkpoint['lr'])

    return model, opt, checkpoint['epoch'],lo,lr_scheduler

#ckp_path = "./UNet-3-CamVid/checkpoint.pt"

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
    #best_model_wts = copy.deepcopy(torch.load("./UNet-3-CamVid/UNet-3_weights.pt"))

    #best_loss= min(loss_history["val"])
    best_loss=float('inf')
    o = open('./UNet-3-CamVid/UNet-3.txt','a')
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
path2models= "./UNet-3-CamVid/UNet-3_"
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
o = open('./UNet-3-CamVid/UNet-3-time.txt','w')

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
plt.savefig('./UNet-3-CamVid/UNet-3.png', dpi = 300)

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
o = open('./UNet-3-CamVid/UNet-3_IOU_val.txt','w')

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
o = open('./UNet-3-CamVid/UNet-3_IOU_test.txt','w')

print(i_sum/233, file=o)

o.close()

a = loss_hist["train"]
A = [int(x) for x in a]
b = loss_hist["val"]
B = [int(x) for x in b]

import csv

with open('./UNet-3-CamVid/UNet-3-plot.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(A,B))





