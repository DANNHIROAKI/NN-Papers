from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
import json
import torchvision.transforms as tvt

from cloc import dataset_paths, register_dataset

__all__ = ['ImageFolder']


class ImageFolder(Dataset):
    def __init__(self, root, transform=tvt.ToTensor()):
        self.root = root # will be accessed by the training script
        self.transform = transform
        # scan and add images
        self.image_paths = sorted(Path(root).rglob('*.*'))
        assert len(self.image_paths) > 0, f'Found {len(self.image_paths)} images in {root}.'

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        impath = self.image_paths[index]
        img = Image.open(impath).convert('RGB') 
        im = self.transform(img)
        return im


@register_dataset
def celeba_hq256train():
    return ImageFolder(root=dataset_paths['celeba'] / 'hq-train-256')

@register_dataset
def celeba_hq256val():
    return ImageFolder(root=dataset_paths['celeba'] / 'hq-val-256')

@register_dataset
def celeba_hq256test():
    return ImageFolder(root=dataset_paths['celeba'] / 'hq-test-256')


@register_dataset
def coco_train2017(crop=256):
    root = dataset_paths['coco'] / 'train2017'
    transform = tvt.Compose([
        tvt.RandomCrop(crop, pad_if_needed=True, padding_mode='reflect'),
        tvt.RandomHorizontalFlip(p=0.5),
        tvt.ToTensor()
    ])
    return ImageFolder(root, transform=transform)


@register_dataset
def kodak():
    return ImageFolder(root=dataset_paths['kodak'])
