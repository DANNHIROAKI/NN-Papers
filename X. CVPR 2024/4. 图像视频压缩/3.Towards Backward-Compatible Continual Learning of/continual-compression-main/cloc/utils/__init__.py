from .coding import *
from .general import *
from .lr_schedulers import *


from PIL import Image
from pathlib import Path
import torchvision.transforms.functional as tvf

def get_file_size(fpath):
    """ Get file size in bits
    """
    return Path(fpath).stat().st_size * 8

def tv_imread(impath):
    img = Image.open(impath).convert('RGB')
    return tvf.to_tensor(img).unsqueeze_(0)
