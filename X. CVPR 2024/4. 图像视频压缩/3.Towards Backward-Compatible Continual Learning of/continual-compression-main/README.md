# Towards Backward-Compatible Continual Learning of Image Compression

This repository contains the authors' implementation of the paper:
> **Towards Backward-Compatible Continual Learning of Image Compression** <br>
> Zhihao Duan, Ming Lu, Justin Yang, Jiangpeng He, Zhan Ma, Fengqing Zhu <br>
> https://arxiv.org/abs/2402.18862 <br>
> CVPR 2024

- [Install](#install)
- [Usage](#usage)
- [Licenses](#license)


## Install
**Requirements**:
- Python
- PyTorch >= 2.0 : https://pytorch.org/get-started/locally
- `python -m pip install tqdm compressai timm`
- [Optional, required for training] `python -m pip install wandb`

**Download and Install**:
1. Download the repository;
2. Modify the dataset paths in `cloc/paths.py`.
3. pip install the repository (in development mode): `python -m pip install -e .`


## Usage
### Pre-training
```
python pre-train.py --model msh_vr
```

### Pre-trained model weights
```python
from cloc import get_model
model = get_model('msh_vr', pretrained=True) # weights will be downloaded automatically
model.eval()
```

### Backward-compatible fine-tuning on a new dataset
```
python train-inc.py
```
The default dataset being used here is CelebA 256x256.


### Evaluation
TBD


## License
TBD
