import math
import os
import cv2
import numpy as np
from functools import partial
from numba import jit
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensor



IMG_SIZE = 256
IMG_SIZE_2 = IMG_SIZE * 2
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
INPUT_PATH = '...'
OUT_DIR = '...'



def fn_lognorm(x):
    return torch.log(255*x + 1)/np.log(256)




def load_dataset(batch_size=32):
    data_path = INPUT_PATH
    global train_dataset
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transforms.Compose([
            #T.Resize((512,512)),
            T.Pad((0, 3, 0, 3)),
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),
            T.Lambda(fn_lognorm),
            T.Normalize(mean=(0.5), std=(0.5), inplace=True)
            #adding gamma correction
        ])
    )
    
    loaders = []
    train_loader = DataLoader(
        train_dataset,
        batch_size = 32,
        shuffle = True,
        drop_last = True,
        num_workers = 16 
    )
    
    loaders.append(train_loader)
    return loaders
