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



    self.s1 = nn.Sequential(nn.Conv2d(64, 12, kernel_size=1, stride =1,bias=False),
                                   nn.BatchNorm2d(12),
                                   nn.ReLU(inplace=True))
    self.pool = nn.AvgPool2d(kernel_size=2)
    
    self.glance21 = look(in_filters=67, dilation_rate=3)
    self.glance22 = look(in_filters=99, dilation_rate=1)
    
    
    
    self.s2 = nn.Sequential(nn.Conv2d(128, 12, kernel_size=1, stride =1,bias=False),
                                   nn.BatchNorm2d(12),
                                  nn.ReLU(inplace=True))
    
    self.glance31 = look(in_filters=131, dilation_rate=2)
    self.glance32 = look(in_filters=163, dilation_rate=3)



    self.s3 = nn.Sequential(nn.Conv2d(192, 12, kernel_size=1, stride =1,bias=False),
                                  nn.BatchNorm2d(12),
                                  nn.ReLU(inplace=True))
 
    self.out = nn.Sequential(nn.Conv2d(36, 12, kernel_size=1, stride =1,bias=False))


    
    
  def forward(self,x):
    R = self.glance11(x)
    c = torch.cat((x,R),dim=1)
    
    R = self.glance12(c)
    c = torch.cat((c,R),dim=1)



    # removing the image
    s1 = self.s1(c[:,3:,:,:])
    
    c = self.pool(c)

    R = self.glance21(c)
    c = torch.cat((c,R),dim=1)
    
    R = self.glance22(c)
    c = torch.cat((c,R),dim=1)



    # removing the image
    s2 = self.s2(c[:,3:,:,:])
    
    s2 =  torch.nn.functional.interpolate(s2, (368,480), mode='bilinear',align_corners=True)

    c = self.pool(c)
    
    R = self.glance31(c)
    c = torch.cat((c,R),dim=1)
    
    R = self.glance32(c)
    c = torch.cat((c,R),dim=1)



    # removing the image
    s3 = self.s3(c[:,3:,:,:])

    s3 =  torch.nn.functional.interpolate(s3, (368,480), mode='bilinear',align_corners=True)
    R = torch.cat((s1,s2,s3),dim=1)
    
    R = self.out(R)

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
    o = open('./Ikshana-3S-2G-1-Camvid-BS2/Ikshana-3S-2G-1.txt','w')
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
path2models= "./Ikshana-3S-2G-1-Camvid-BS2/Ikshana-3S-2G-1_"
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
o = open('./Ikshana-3S-2G-1-Camvid-BS2/Ikshana-3S-2G-1-time.txt','w')

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
plt.savefig('./Ikshana-3S-2G-1-Camvid-BS2/Ikshana-3S-2G-1.png', dpi = 300)

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
o = open('./Ikshana-3S-2G-1-Camvid-BS2/Ikshana-3S-2G-1_IOU_val.txt','w')

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
o = open('./Ikshana-3S-2G-1-Camvid-BS2/Ikshana-3S-2G-1_IOU_test.txt','w')

print(i_sum/233, file=o)

o.close()

a = loss_hist["train"]
A = [int(x) for x in a]
b = loss_hist["val"]
B = [int(x) for x in b]

import csv

with open('./Ikshana-3S-2G-1-Camvid-BS2/plot.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(A,B))





