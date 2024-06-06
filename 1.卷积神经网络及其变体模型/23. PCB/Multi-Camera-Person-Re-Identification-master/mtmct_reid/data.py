from pathlib import Path
from typing import Optional, Union
from typing_extensions import Literal

import joblib
import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.dataset import Subset
from torchvision import transforms

from .metrics import smooth_st_distribution
from .utils import get_ids


class ReIDDataset(Dataset):
    """
    The ReID Dataset module is a custom Dataset module, specific to parsing 
        the Market & Duke Person Re-Identification datasets.

    Args:
        data_dir (str): The path where the dataset is located.

        transform ([list, torchvision.transforms], optional): Pass the list of
            transforms to transform the input images. Defaults to None.

        target_transform ([list, torchvision.transforms], optional): Pass the
            list of transforms to transform the labels. Defaults to None.

        ret_camid_n_frame (bool, optional): Whether to return camera ids and
            frames. True will additionally return cam_ids and frame. 
            Defaults to False.

    Raises:
        Exception: If directory does not exist!

    """

    def __init__(self, data_dir: str,
                 dataset: Literal['market', 'duke'] = 'market', transform=None,
                 target_transform=None,
                 ret_camid_n_frame: bool = False):

        super(ReIDDataset, self).__init__()
        self.data_dir = Path(data_dir)

        if not self.data_dir.exists():
            raise Exception(
                f"'Path '{self.data_dir.__str__()}' does not exist!")
        if not self.data_dir.is_dir():
            raise Exception(
                f"Path '{self.data_dir.__str__()}' is not a directory!")

        self.transform = transform
        self.target_transform = target_transform
        self.ret_camid_n_frame = ret_camid_n_frame
        self.dataset = dataset
        self._init_data()

    def _init_data(self):

        self.imgs = list(self.data_dir.glob('*.jpg'))
        # Filter out labels with -1
        self.imgs = [img for img in self.imgs if '-1' not in img.stem]

        self.cam_ids, self.labels, self.frames = get_ids(
            self.imgs, self.dataset)

        self.num_cams = len(set(self.cam_ids))
        self.classes = tuple(set(self.labels))

        # Convert labels to continuous idxs
        self.class_to_idx = {label: i for i, label in enumerate(self.classes)}
        self.targets = [self.class_to_idx[label] for label in self.labels]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        sample = Image.open(str(self.imgs[index])).convert('RGB')
        target = self.targets[index]

        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            target = self.target_transform(target)

        if self.ret_camid_n_frame:
            cam_id = self.cam_ids[index]
            frame = self.frames[index]
            return sample, target, cam_id, frame

        return sample, target


class ReIDDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str,
                 dataset: Literal['market', 'duke'] = 'market',
                 st_distribution: Optional[str] = None,
                 train_subdir: str = 'bounding_box_train',
                 test_subdir: str = 'bounding_box_test',
                 query_subdir: str = 'query', train_batchsize: int = 16,
                 val_batchsize: int = 16, test_batchsize: int = 16,
                 num_workers: int = 4,
                 random_erasing: float = 0.0, color_jitter: bool = False,
                 save_distribution: Union[bool, str] = False):

        super().__init__()

        self.data_dir = Path(data_dir)
        self.dataset = dataset

        if not self.data_dir.exists():
            raise Exception(
                f"'Path '{self.data_dir.__str__()}' does not exist!")
        if not self.data_dir.is_dir():
            raise Exception(
                f"Path '{self.data_dir.__str__()}' is not a directory!")

        self.train_dir = self.data_dir / train_subdir
        self.test_dir = self.data_dir / test_subdir
        self.query_dir = self.data_dir / query_subdir

        self.train_batchsize = train_batchsize
        self.test_batchsize = test_batchsize
        self.val_batchsize = val_batchsize
        self.num_workers = num_workers

        self.color_jitter = color_jitter
        self.random_erasing = random_erasing

        self.st_distribution = st_distribution
        self.save_distribution = save_distribution

        self.prepare_data()

    def prepare_data(self):

        train_transforms = [transforms.Resize((384, 192), interpolation=3),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [
                                0.229, 0.224, 0.225])
                            ]
        test_transforms = train_transforms
        test_transforms.pop(1)

        if self.random_erasing > 0:
            train_transforms.append(
                transforms.RandomErasing(self.random_erasing))
        if self.color_jitter:
            train_transforms.append(transforms.ColorJitter())

        train_transforms = transforms.Compose(train_transforms)
        self.train = ReIDDataset(
            self.train_dir, self.dataset, train_transforms)
        self.num_classes = len(self.train.classes)
        train_len = int(len(self.train) * 0.8)
        test_len = len(self.train) - train_len
        self.train, self.test = random_split(self.train, [train_len, test_len])

        test_transforms = transforms.Compose(test_transforms)
        self.test = ReIDDataset(self.test_dir, self.dataset, test_transforms)
        self.query = ReIDDataset(self.query_dir, self.dataset, test_transforms,
                                 ret_camid_n_frame=True)
        self.gallery = ReIDDataset(self.test_dir, self.dataset, test_transforms,
                                   ret_camid_n_frame=True)

        self._load_st_distribution()
        if self.save_distribution:
            self._save_st_distribution()

    def _load_st_distribution(self):

        if isinstance(self.st_distribution, str):
            self.st_distribution = Path(self.st_distribution)

            if not (self.st_distribution.exists()
                    and self.st_distribution.is_file()):
                raise FileNotFoundError(
                    f"Location '{str(self.st_distribution)}' \
                    does not exist or not a file!")

            if self.st_distribution.suffix != '.pkl':
                raise ValueError('File must be of type .pkl')

            print(
                f'\nLoading Spatial-Temporal Distribution from \
                {self.st_distribution}.\n\n')
            self.st_distribution = joblib.load(str(self.st_distribution))

        elif not self.st_distribution:
            print('\n\nGenerating Spatial-Temporal Distribution.\n\n')
            num_cams = self.query.num_cams
            max_hist = 5000 if self.query.dataset == 'market' else 3000

            cam_ids = self.query.cam_ids + self.gallery.cam_ids
            targets = self.query.targets + self.gallery.targets
            frames = self.query.frames + self.gallery.frames

            self.st_distribution = smooth_st_distribution(cam_ids, targets,
                                                          frames,
                                                          num_cams, max_hist)

    def _save_st_distribution(self):
        if isinstance(self.save_distribution, str):
            if '.pkl' not in self.save_distribution:
                self.save_distribution += '.pkl'
        else:
            self.save_distribution = self.data_dir + 'st_distribution.pkl'

        print(
            f'\nSaving distribution at {self.save_distribution}')
        joblib.dump(self.st_distribution, self.save_distribution)

    def train_dataloader(self):

        return DataLoader(self.train, batch_size=self.train_batchsize,
                          shuffle=True, num_workers=self.num_workers,
                          pin_memory=True)

    def val_dataloader(self):
        test_loader = DataLoader(self.test, batch_size=self.test_batchsize,
                                 shuffle=False, num_workers=self.num_workers,
                                 pin_memory=True)
        query_indices = range(self.test_batchsize)
        query_loader = DataLoader(Subset(self.query, query_indices),
                                  batch_size=self.test_batchsize,
                                  shuffle=False, num_workers=self.num_workers,
                                  pin_memory=True)
        evens = list(range(0, len(self.gallery), 3))
        gall_loader = DataLoader(Subset(self.gallery, evens),
                                 batch_size=self.test_batchsize,
                                 shuffle=True, num_workers=self.num_workers,
                                 pin_memory=True)

        return [query_loader, gall_loader, test_loader]

    def test_dataloader(self):

        query_loader = DataLoader(self.query, batch_size=self.test_batchsize,
                                  shuffle=False, num_workers=self.num_workers,
                                  pin_memory=True)
        gall_loader = DataLoader(self.gallery, batch_size=self.test_batchsize,
                                 shuffle=True, num_workers=self.num_workers,
                                 pin_memory=True)

        return [query_loader, gall_loader]
