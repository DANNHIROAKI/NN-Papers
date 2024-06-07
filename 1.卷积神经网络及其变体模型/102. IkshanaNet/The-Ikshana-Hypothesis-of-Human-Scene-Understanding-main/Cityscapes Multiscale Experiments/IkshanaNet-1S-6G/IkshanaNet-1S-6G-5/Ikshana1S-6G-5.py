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
        
val = myCityscapes("./", split='val', mode='fine', target_type='semantic', transforms=transform_val, target_transform=None)

train , dump = torch.utils.data.random_split(train_ds, [92,2883], generator=torch.Generator().manual_seed(42))

print(len(train))
print(len(val))
print(len(dump))

#defining Dataloaders
from torch.utils.data import DataLoader
train_dl = DataLoader(train, batch_size=2, shuffle=True)
val_dl = DataLoader(val, batch_size=2, shuffle=False)

class look(nn.Module):
  def __init__(self, in_filters, dilation_rate):
    super(look,self).__init__()
    self.conv_main = nn.Sequential(nn.Conv2d(in_filters, 32, kernel_size=3, stride =1,padding=dilation_rate, dilation=dilation_rate, bias=False),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(32, 32, kernel_size=3, stride =1,padding=dilation_rate, dilation=dilation_rate, bias=False),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(32, 32, kernel_size=3, stride =1,padding=dilation_rate, dilation=dilation_rate, bias=False),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True))
    

  def forward(self,x):
    
    x = self.conv_main(x)

    return x


  
class IkshanaNet(nn.Module):
  def __init__(self):
    super(IkshanaNet,self).__init__()
    self.glance11 = look(in_filters=3, dilation_rate=1)
    self.glance12 = look(in_filters=35, dilation_rate=2)
    self.glance13 = look(in_filters=67, dilation_rate=3)
    self.glance14 = look(in_filters=99, dilation_rate=1)
    self.glance15 = look(in_filters=131, dilation_rate=2)
    self.glance16 = look(in_filters=163, dilation_rate=3)

 
    self.out = nn.Sequential(nn.Conv2d(192, 20, kernel_size=1, stride =1,bias=False))


    
    
  def forward(self,x):
    R = self.glance11(x)
    c = torch.cat((x,R),dim=1)
    
    R = self.glance12(c)
    c = torch.cat((c,R),dim=1)

    R = self.glance13(c)
    c = torch.cat((c,R),dim=1)

    R = self.glance14(c)
    c = torch.cat((c,R),dim=1)

    R = self.glance15(c)
    c = torch.cat((c,R),dim=1)
    
    R = self.glance16(c)
    c = torch.cat((c,R),dim=1)
    

    R = self.out(c[:,3:,:,:])

    return R

    



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
    o = open('./IkshanaNet-1S-6G-5/IkshanaNet-1S-6G-5-Trainingset.txt','w')
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
path2models= "./IkshanaNet-1S-6G-5/IkshanaNet-1S-6G-5-Trainingset_"
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
o = open('./IkshanaNet-1S-6G-5/IkshanaNet-1S-6G-5_time.txt','w')

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
plt.savefig('./IkshanaNet-1S-6G-5/IkshanaNet-1S-6G-5.png', dpi = 300)


a = loss_hist["train"]
A = [int(x) for x in a]
b = loss_hist["val"]
B = [int(x) for x in b]

import csv

with open('./IkshanaNet-1S-6G-5/plot.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(A,B))

