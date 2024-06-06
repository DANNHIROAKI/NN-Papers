'''
This file contains the global setting of dataset paths.
'''
from pathlib import Path


# The root directory of all datasets
_root = Path('~/datasets').expanduser().resolve()

dataset_paths = {
    # Kodak images: http://r0k.us/graphics/kodak
    'kodak': _root / 'kodak',

    # CLIC challenge and test sets: http://www.compression.cc
    'clic2022-test': _root / 'clic/test-2022',

    # Tecnick TESTIMAGES: https://testimages.org
    'tecnick-rgb-1200': _root / 'tecnick/TESTIMAGES/RGB/RGB_OR_1200x1200',

    # COCO: https://cocodataset.org
    'coco': _root / 'coco',

    # CelebA: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
    'celeba': _root / 'celeba',
}
