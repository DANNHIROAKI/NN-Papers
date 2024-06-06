""" load .txt to generate jpg picture. """
import os
import datetime
import numpy as np
from PIL import Image

if __name__ == "__main__":

    imageSize = 64
    nc = 3

    f_name = os.path.join("../data/mxbase_result/result.txt")

    fake = np.loadtxt(f_name, np.float32).reshape(1, nc, imageSize, imageSize)

    img_pil = fake[0, ...].reshape(1, nc, imageSize, imageSize)
    img_pil = img_pil[0].astype(np.uint8).transpose((1, 2, 0))
    img_pil = Image.fromarray(img_pil)
    now_time = datetime.datetime.now()
    img_name = str(now_time.hour) + str(now_time.minute) + str(now_time.second)
    img_pil.save(os.path.join("../data/mxbase_result/", "generated_%s.png") % img_name)

    print("Generate images success!")
