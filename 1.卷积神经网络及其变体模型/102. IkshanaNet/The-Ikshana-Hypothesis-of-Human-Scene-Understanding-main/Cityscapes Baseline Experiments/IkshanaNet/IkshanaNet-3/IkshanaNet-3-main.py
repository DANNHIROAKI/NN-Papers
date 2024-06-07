import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.utils.model_zoo as model_zoo
import numpy as np
import sys, time, os, warnings 
from skimage.segmentation import mark_boundaries
import matplotlib.pylab as plt
#%matplotlib inline
from sklearn.model_selection import train_test_split
from torchvision.datasets import Cityscapes

from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image
from albumentations import (Compose,Resize,Normalize)

#mean = [0.28689554, 0.32513303, 0.28389177]
#std = [0.18696375, 0.19017339, 0.18720214]
mean = [0.485, 0.456, 0.406] 
std = [0.229, 0.224, 0.225]
h,w=512,1024

transform_train = Compose([ Resize(h,w), 
                Normalize(mean=mean,std=std)])

transform_val = Compose( [ Resize(h,w),
                          Normalize(mean=mean,std=std)])
class myCityscapes(Cityscapes):
    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')

        targets = []
        for i, t in enumerate(self.target_type):
            if t == 'polygon':
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])

            targets.append(target)

        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.transforms is not None:
            sample= self.transforms(image=np.array(image), mask=np.array(target))
#            sample = self.transform(**sample)
            img = sample['image']
            target = sample['mask'] 
            #img, mask = self.transforms(np.array(image),np.array(target))
            
        img = to_tensor(img)
        mask = torch.from_numpy(target).type(torch.long)

        return img, mask
    
    def _get_target_suffix(self, mode, target_type):
            
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelTrainIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        else:
            return '{}_polygons.json'.format(mode)
        
        
train_ds = myCityscapes("./", split='train', mode='fine',target_type='semantic', transforms=transform_train,target_transform=None)
           #transforms=None)
val = myCityscapes("./", split='val', mode='fine', target_type='semantic', transforms=transform_val, target_transform=None)
#           transforms=None)

train , dump = torch.utils.data.random_split(train_ds, [743,2232], generator=torch.Generator().manual_seed(42))

print(len(train))
print(len(dump))
print(len(val))

#defining Dataloaders
from torch.utils.data import DataLoader
train_dl = DataLoader(train, batch_size=2, shuffle=True)
val_dl = DataLoader(val, batch_size=2, shuffle=False)

class IkshanaNet(nn.Module):
    
  def __init__(self):
    super(IkshanaNet,self).__init__()
    self.conv11 =  nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, stride =1,padding=1, dilation=1, bias=False),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(32, 32, kernel_size=3, stride =1,padding=1, dilation=1, bias=False),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(32, 32, kernel_size=3, stride =1,padding=1, dilation=1, bias=False),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True))
    self.conv12 = nn.Sequential(nn.Conv2d(35, 32, kernel_size=3, stride =1,padding=2, dilation=2, bias=False),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(32, 32, kernel_size=3, stride =1,padding=2, dilation=2, bias=False),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(32, 32, kernel_size=3, stride =1,padding=2, dilation=2, bias=False),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True))
    self.conv13 = nn.Sequential(nn.Conv2d(67, 32, kernel_size=3, stride =1,padding=3, dilation=3, bias=False),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(32, 32, kernel_size=3, stride =1,padding=3, dilation=3, bias=False),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(32, 32, kernel_size=3, stride =1,padding=3, dilation=3, bias=False),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True))
    self.conv14 = nn.Sequential(nn.Conv2d(96, 96, kernel_size=3, stride =1,padding=1,bias=False),
                                nn.BatchNorm2d(96),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(96, 96, kernel_size=3, stride =1,padding=1,bias=False),
                                nn.BatchNorm2d(96),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(96, 96, kernel_size=3, stride =1,padding=1,bias=False),
                                nn.BatchNorm2d(96),
                                nn.ReLU(inplace=True))
    
    self.pool1 = nn.AvgPool2d(kernel_size=2)
    
    self.side11 = nn.Sequential(nn.Conv2d(96, 20, kernel_size=1, stride =1,bias=False),
                                nn.BatchNorm2d(20),
                                nn.ReLU(inplace=True))
    
    self.conv21 = nn.Sequential(nn.Conv2d(99, 32, kernel_size=3, stride =1,padding = 1, dilation =1,bias=False),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(32, 32, kernel_size=3, stride =1,padding = 1, dilation =1,bias=False),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(32, 32, kernel_size=3, stride =1,padding = 1, dilation =1,bias=False),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True))
    self.conv22 = nn.Sequential(nn.Conv2d(131, 32, kernel_size=3, stride =1,padding=2, dilation=2, bias=False),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(32, 32, kernel_size=3, stride =1,padding=2, dilation=2, bias=False),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(32, 32, kernel_size=3, stride =1,padding=2, dilation=2, bias=False),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True))
    self.conv23 = nn.Sequential(nn.Conv2d(163, 32, kernel_size=3, stride =1,padding=3, dilation=3, bias=False),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(32, 32, kernel_size=3, stride =1,padding=3, dilation=3, bias=False),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(32, 32, kernel_size=3, stride =1,padding=3, dilation=3, bias=False),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True))
    self.conv24 = nn.Sequential(nn.Conv2d(192, 192, kernel_size=3, stride =1,padding=1,bias=False),
                                nn.BatchNorm2d(192),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(192, 192, kernel_size=3, stride =1,padding=1,bias=False),
                                nn.BatchNorm2d(192),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(192, 192, kernel_size=3, stride =1,padding=1,bias=False),
                                nn.BatchNorm2d(192),
                                nn.ReLU(inplace=True))
    self.pool2 = nn.AvgPool2d(kernel_size=2)
    
    self.side21 = nn.Sequential(nn.Conv2d(192, 20, kernel_size=1, stride =1,bias=False),
                                nn.BatchNorm2d(20),
                                nn.ReLU(inplace=True))
                                   
    
    self.conv31 = nn.Sequential(nn.Conv2d(195, 32, kernel_size=3, stride =1,padding = 1, dilation =1,bias=False),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(32, 32, kernel_size=3, stride =1,padding = 1, dilation =1,bias=False),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(32, 32, kernel_size=3, stride =1,padding = 1, dilation =1,bias=False),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True))
    self.conv32 = nn.Sequential(nn.Conv2d(227, 32, kernel_size=3, stride =1,padding=2, dilation=2, bias=False),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(32, 32, kernel_size=3, stride =1,padding=2, dilation=2, bias=False),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(32, 32, kernel_size=3, stride =1,padding=2, dilation=2, bias=False),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True))
    self.conv33 = nn.Sequential(nn.Conv2d(259, 32, kernel_size=3, stride =1,padding=3, dilation=3, bias=False),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(32, 32, kernel_size=3, stride =1,padding=3, dilation=3, bias=False),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(32, 32, kernel_size=3, stride =1,padding=3, dilation=3, bias=False),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True))
    self.conv34 = nn.Sequential(nn.Conv2d(288, 288, kernel_size=3, stride =1,padding=1,bias=False),
                                nn.BatchNorm2d(288),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(288, 288, kernel_size=3, stride =1,padding=1,bias=False),
                                nn.BatchNorm2d(288),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(288, 288, kernel_size=3, stride =1,padding=1,bias=False),
                                nn.BatchNorm2d(288),
                                nn.ReLU(inplace=True))

    self.side31 = nn.Sequential(nn.Conv2d(288, 20, kernel_size=1, stride =1,bias=False),
                                nn.BatchNorm2d(20),
                                nn.ReLU(inplace=True))
                               
 
    self.out = nn.Sequential(nn.Conv2d(60, 20, kernel_size=1, stride =1,bias=False))


  def forward(self, x):
    x1  = self.conv11(x) 
    c1  = torch.cat((x,x1),dim=1)
    x1  = self.conv12(c1)
    c1  = torch.cat((c1,x1),dim=1)
    x1  = self.conv13(c1)
    c1  = torch.cat((c1,x1),dim=1)
    #Removing the Image from the feature map
    x1  = c1[:,3:,:,:]
    #a 3X3 conv projection layer, that takes and outputs the same number of filters
    x1  = self.conv14(x1)

    s1  = self.side11(x1)


    x2  = self.pool1(x1)
    #Re size original imgae to half 
    x2i = torch.nn.functional.interpolate(x, (256,512), mode='bilinear',align_corners=True)
    c2  = torch.cat((x2i,x2),dim=1)
    x2  = self.conv21(c2)
    c2  = torch.cat((c2,x2),dim=1)
    x2  = self.conv22(c2)
    c2  = torch.cat((c2,x2),dim=1)
    x2  = self.conv23(c2)
    c2  = torch.cat((c2,x2),dim=1)
    #Removing the Image from the feature map
    x2  = c2[:,3:,:,:]
    #a 3X3 conv projection layer, that takes and outputs same number of filters
    x2  = self.conv24(x2)

    s2  = self.side21(x2)

    x3  = self.pool2(x2)
    #Re Size original image to quater'th size
    x3i = torch.nn.functional.interpolate(x, (128,256), mode='bilinear',align_corners=True)
    c3  = torch.cat((x3i,x3),dim=1)

    x3  = self.conv31(c3)
    c3  = torch.cat((c3,x3),dim=1)
    x3  = self.conv32(c3)
    c3  = torch.cat((c3,x3),dim=1)
    x3  = self.conv33(c3)
    c3  = torch.cat((c3,x3),dim=1)
    #Removing the Imgae from the feature map
    x3  = c3[:,3:,:,:]
    x3  = self.conv34(x3)

    s3  = self.side31(x3)
        
    s3 = torch.nn.functional.interpolate(s3, (256,512), mode='bilinear',align_corners=True)
    s3 = torch.cat((s3,s2),dim=1)
    s3 = torch.nn.functional.interpolate(s3, (512,1024), mode='bilinear',align_corners=True)
    s3 = torch.cat((s3,s1),dim=1)

    s3 = self.out(s3)

    return s3
    

model = IkshanaNet()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model=model.to(device)

#criterion = nn.BCEWithLogitsLoss(reduction="sum")
criterion = nn.CrossEntropyLoss(reduction="sum")
from torch import optim
opt = optim.SGD(model.parameters(), lr=1e-6, momentum=0.7,nesterov=True)

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

    loss_history={
        "train": [],
        "val": []}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss=float('inf')
    o = open('./IkshanaNet-Quarter-Trainingset.txt','w')
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
    model.load_state_dict(best_model_wts)
    o.close()
    return model, loss_history



start = time.time()

import os
path2models= "./IkshanaNet-Quarter-Trainingset_"
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
o = open('./IkshanaNet-Quarter-Trainingset_time.txt','w')

print("TIME TOOK {:3.2f}MIN".format((end - start )/60), file=o)

o.close()

print("TIME TOOK {:3.2f}MIN".format((end - start )/60))

torch.save(model.state_dict(), "IkshanaNet-Quarter-Trainingset_M")

num_epochs=params_train["num_epochs"]
plt.figure(figsize=(30,30))
plt.title("Train-Val Loss")
plt.plot(range(1,num_epochs+1),loss_hist["train"],label="train")
plt.plot(range(1,num_epochs+1),loss_hist["val"],label="val")
plt.ylabel("Loss")
plt.xlabel("Training Epochs")
plt.legend()
plt.savefig('IkshanaNet-Quarter-Trainingset.png', dpi = 300)


