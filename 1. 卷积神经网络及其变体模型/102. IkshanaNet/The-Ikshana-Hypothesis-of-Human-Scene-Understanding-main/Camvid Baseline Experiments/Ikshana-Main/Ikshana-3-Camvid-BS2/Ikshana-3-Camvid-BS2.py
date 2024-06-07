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
    
    self.side11 = nn.Sequential(nn.Conv2d(96, 12, kernel_size=1, stride =1,bias=False),
                                nn.BatchNorm2d(12),
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
    
    self.side21 = nn.Sequential(nn.Conv2d(192, 12, kernel_size=1, stride =1,bias=False),
                                nn.BatchNorm2d(12),
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

    self.side31 = nn.Sequential(nn.Conv2d(288, 12, kernel_size=1, stride =1,bias=False),
                                nn.BatchNorm2d(12),
                                nn.ReLU(inplace=True))
                               
 
    self.out = nn.Sequential(nn.Conv2d(36, 12, kernel_size=1, stride =1,bias=False))


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
    x2i = torch.nn.functional.interpolate(x, (184,240), mode='bilinear',align_corners=True)
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
    x3i = torch.nn.functional.interpolate(x, (92,120), mode='bilinear',align_corners=True)
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
        
    s3 = torch.nn.functional.interpolate(s3, (184,240), mode='bilinear',align_corners=True)
    s3 = torch.cat((s3,s2),dim=1)
    s3 = torch.nn.functional.interpolate(s3, (368,480), mode='bilinear',align_corners=True)
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
    o = open('./Ikshana-3-Camvid-BS2/Ikshana-3.txt','w')
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
path2models= "./Ikshana-3-Camvid-BS2/Ikshana-3_"
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
o = open('./Ikshana-3-Camvid-BS2/Ikshana-3-time.txt','w')

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
plt.savefig('./Ikshana-3-Camvid-BS2/Ikshana-3.png', dpi = 300)

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
o = open('./Ikshana-3-Camvid-BS2/Ikshana-3_IOU_val.txt','w')

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
o = open('./Ikshana-3-Camvid-BS2/Ikshana-3_IOU_test.txt','w')

print(i_sum/233, file=o)

o.close()

a = loss_hist["train"]
A = [int(x) for x in a]
b = loss_hist["val"]
B = [int(x) for x in b]

import csv

with open('./Ikshana-3-Camvid-BS2/plot.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(A,B))







