""" dataset """
import os
import mindspore.dataset as ds
import mindspore.dataset.vision as c
import mindspore.dataset.transforms as C2
import mindspore.common.dtype as mstype
from mindspore.communication.management import get_rank, get_group_size

def _get_rank_info():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if rank_size > 1:
        rank_size = get_group_size()
        rank_id = get_rank()
    else:
        rank_size = 1
        rank_id = 0

    return rank_size, rank_id

def create_dataset(dataroot, dataset, batchSize, imageSize, repeat_num=1, workers=8, target='Ascend'):
    """Create dataset"""

    device_num, rank_id = _get_rank_info()
    # define map operations
    resize_op = c.Resize(imageSize)
    center_crop_op = c.CenterCrop(imageSize)
    normalize_op = c.Normalize(mean=(0.5*255, 0.5*255, 0.5*255), std=(0.5*255, 0.5*255, 0.5*255))
    hwc2chw_op = c.HWC2CHW()

    if dataset == 'lsun':
        if device_num == 1:
            data_set = ds.ImageFolderDataset(dataroot, num_parallel_workers=workers, shuffle=True, decode=True)
        else:
            data_set = ds.ImageFolderDataset(dataroot, num_parallel_workers=workers, shuffle=True, decode=True,
                                             num_shards=device_num, shard_id=rank_id)

        transform = [resize_op, center_crop_op, normalize_op, hwc2chw_op]
    else:
        if device_num == 1:
            data_set = ds.Cifar10Dataset(dataroot, num_parallel_workers=workers, shuffle=True)
        else:
            data_set = ds.Cifar10Dataset(dataroot, num_parallel_workers=workers, shuffle=True, \
                                         num_shards=device_num, shard_id=rank_id)

        transform = [resize_op, normalize_op, hwc2chw_op]

    type_cast_op = C2.TypeCast(mstype.int32)

    data_set = data_set.map(input_columns='image', operations=transform, num_parallel_workers=workers)
    data_set = data_set.map(input_columns='label', operations=type_cast_op, num_parallel_workers=workers)

    data_set = data_set.batch(batchSize, drop_remainder=True)
    data_set = data_set.repeat(repeat_num)

    return data_set
