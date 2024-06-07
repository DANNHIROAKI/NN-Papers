# Learning Single Camera Depth Estimation using Dual-Pixels

Reference code for the paper [Learning Single Camera Depth Estimation using Dual-Pixels](https://arxiv.org/abs/1904.05822).
Rahul Garg, Neal Wadhwa, Sameer Ansari & Jonathan T. Barron, ICCV 2019. If you use this code or our dataset, please cite our paper:
```
@article{GargDualPixelsICCV2019,
  author    = {Rahul Garg and Neal Wadhwa and Sameer Ansari and Jonathan T. Barron},
  title     = {Learning Single Camera Depth Estimation using Dual-Pixels},
  journal   = {ICCV},
  year      = {2019},
}
```

If you have any questions about our app, dataset or code, please email
<dual-pixel-questions@google.com>. We are more likely to respond if you email us
than if you open an issue on GitHub.


## Dataset

<div style="text-align:center"><img src="https://lh3.googleusercontent.com/00X4nO0xwOgB8nnHhj8VNC0tng0q2D3l41ibtNemxiMKJA-eS4xMgNOyahQz7NtscH5xQ5MXVk1nQ3qAbgweiS6FBn5gnPHjOyDGiGx8bH5UqmaThcwHRf-eRCtdgdMyIi76fiksHCg=w2400"/></div>

The dataset containing the RGB images, dual-pixel images, and the depth maps
can be downloaded from the links below:

[Train (95G)](https://storage.googleapis.com/iccv2019-data/train.tgz)

[Test (29G)](https://storage.googleapis.com/iccv2019-data/test.tgz)

The dataset linked above is slightly smaller than the one used in the paper.
Please see the dataset [README](https://storage.googleapis.com/iccv2019-data/README.pdf) for more details about the dataset.

## Results and Evaluation.

### Results
Since the dataset above is slightly smaller than the one used in the paper,
we trained and evaluated our best performing model (DPNet with Affine
Invariance) on the data above. The metrics are similar to those reported in the
paper:

![$ \mathrm{AIWE}(1) = 0.0181 \quad \mathrm{AIWE}(2) = 0.0268 \quad 1 - |\rho_s| = 0.152 $](https://render.githubusercontent.com/render/math?math=%24%20%5Cmathrm%7BAIWE%7D(1)%20%3D%200.0181%20%5Cquad%20%5Cmathrm%7BAIWE%7D(2)%20%3D%200.0268%20%5Cquad%201%20-%20%7C%5Crho_s%7C%20%3D%200.152%20%24)

Predictions from the model corresponding to the center image in the test dataset
are available [here](https://storage.googleapis.com/iccv2019-data/model_prediction.tgz) as EXR images or binary numpy files.

### Evaluation

The python script "script.py" in the "eval" directory can be used to evaluate
the predictions. Assuming that the test dataset is in the "test" folder and
predictions are in the "model_prediction" folder, evaluation can be run as:

```
python -m dual_pixels.eval.script --test_dir=test --prediction_dir=model_prediction
```
This has been tested with Python 3.

## Continuous-valued Depth Maps

The depth maps used in the paper and shared above are computed using plane sweep algorithm with 256 planes. Hence, the resulting depth maps are quantized to 256 levels. We also share continuous-valued depth maps corresponding to the [train](https://storage.googleapis.com/iccv2019-data/train_continuous_depth.tgz) and [test](https://storage.googleapis.com/iccv2019-data/test_continuous_depth.tgz) datasets. They use the same depth sampling as described in the [README](https://storage.googleapis.com/iccv2019-data/README.pdf) but are stored as floating point EXR images in the range \[0, 1\]. These depth maps are computed by the algorithm described in [Taniai et. al., Continuous 3D Label Stereo Matching using
Local Expansion Moves](https://taniai.space/projects/stereo/) where the matching cost is computed using VGG features as described in [Guo et. al., The Relightables:
Volumetric Performance Capture of Humans with Realistic Relighting](https://augmentedperception.github.io/therelightables/).


## Android App to Capture Dual-pixel Data

The app has been tested on the Google Pixel 3, Pixel 3 XL, Pixel 4 and Pixel 4 XL.

### Installation instructions:

1. Download [Android Studio](https://developer.android.com/studio). When you install it, make sure to also install the Android SDK API 29.
2. Click "Open an existing Android Studio project". Select the "dual_pixels" directory.
3. There will be a popup with title "Gradle Sync" complaining about a missing file called gradle-wrapper.properties. Click ok to recreate the Gradle wrapper.
4. Plug in your Pixel smartphone. You'll need to enable USB debugging. See
https://developer.android.com/studio/debug/dev-options for further instructions.
5. Go to the "Run" menu at the top and click "Run 'app'" to compile and install the app.

Dual-pixel data will be saved in the directory:
```
/sdcard/Android/data/com.google.reseach.pdcapture/files
```

### Information about saved data

The images are captured with a white level of 1023 (10 bits), but the app
linearly scales them to have white level 65535. That is, the pixels are
initially in the range \[0, 1023\], but are scaled to the range \[0, 65535\].
These images are in linear space. That is, they have not been gamma-encoded. The
black level is typically around 64 before scaling and 4096 after scaling. The
exact black level can be obtained via [SENSOR_DYNAMIC_BLACK_LEVEL](https://developer.android.com/reference/android/hardware/camera2/CaptureResult.html#SENSOR_DYNAMIC_BLACK_LEVEL) in the CaptureResult.

This app only saves the dual-pixel images. It is also possible to save files in
ImageFormat.RAW10 and metadata captured at the same time. Other image stream
combinations may not work.

This app will not work on non-Google phones and will not work on any phone
released by Google prior to the Pixel 3. It should work with the 3a and 4
running Android 10, but is not guaranteed to work on any Pixel phones after that
point.


## Related Work
Wadhwa et al., [Synthetic Depth-of-Field with a Single-Camera Mobile Phone](https://arxiv.org/abs/1806.04171),
SIGGRAPH 2018


