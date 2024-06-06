from PIL import Image
import pickle
import torch
import torch.nn as nn
import torchvision.transforms.functional as tvf


class BaseCompressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('_dummy', torch.zeros(1), persistent=False)
        self._dummy: torch.Tensor

    def preprocess(self, im: torch.Tensor):
        assert im.dim() == 4 and im.shape[1] == 3, f'Expect (b,3,h,w) image input. Got {im.shape=}'
        assert 0.0 <= im.min() and im.max() <= 1.0, f'RGB image {im.min()=}, {im.max()=}'
        x = im.clone().sub_(0.5).mul_(2.0)
        return x

    @torch.inference_mode()
    def postprocess(self, x: torch.Tensor):
        im = x.detach().clone().mul_(0.5).add_(0.5).clamp_(min=0.0, max=1.0)
        return im

    def prepare_compression(self):
        raise NotImplementedError()

    def compress(self, im):
        raise NotImplementedError()

    def decompress(self, compressed_obj):
        raise NotImplementedError()

    def compress_file(self, img_path: str, output_path):
        # read image
        img = Image.open(img_path)
        # TODO: pad image to be divisible by max_stride
        # img_padded = coding.pad_divisible_by(img, div=self.max_stride)
        im = tvf.to_tensor(img).unsqueeze_(0).to(device=self._dummy.device)
        # compress by model
        compressed_obj = self.compress(im)
        # save to file
        with open(output_path, 'wb') as f:
            pickle.dump(compressed_obj, file=f)

    def decompress_file(self, bits_path: str):
        # read from file
        with open(bits_path, 'rb') as f:
            compressed_obj = pickle.load(file=f)
        # TODO: handle padding
        # img_h, img_w = compressed_obj.pop()
        # decompress by model
        im_hat = self.decompress(compressed_obj)
        return im_hat
        # return im_hat[:, :, :img_h, :img_w]
