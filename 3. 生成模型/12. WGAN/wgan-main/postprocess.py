""" postprocess """
import os
import json
import numpy as np
from PIL import Image
from src.args import get_args


if __name__ == "__main__":
    args_opt = get_args('post310')

    with open(args_opt.config, 'r') as gencfg:
        generator_config = json.loads(gencfg.read())

    imageSize = generator_config["imageSize"]
    nc = generator_config["nc"]

    f_name = os.path.join(args_opt.post_result_path, "wgan_bs" + str(args_opt.nimages) + "_0.bin")

    fake = np.fromfile(f_name, np.float32).reshape(args_opt.nimages, nc, imageSize, imageSize)
    fake = np.multiply(fake, 0.5*255)
    fake = np.add(fake, 0.5*255)

    for i in range(args_opt.nimages):
        img_pil = fake[i, ...].reshape(1, nc, imageSize, imageSize)
        img_pil = img_pil[0].astype(np.uint8).transpose((1, 2, 0))
        img_pil = Image.fromarray(img_pil)
        img_pil.save(os.path.join(args_opt.output_dir, "generated_%02d.png" % i))

    print("Generate images success!")
