# Residual Denoising Diffusion Models

This repository is the official implementation of Residual Denoising Diffusion Models.

<p align="center">
<a href="https://github.com/nachifur/RDDM" target="_blank">
<img width="800" height="400" img align="center" alt="RDDM" src="https://github.com/nachifur/RDDM/blob/main/poster/Jiawei_9969.png" />
</a>
</p>

Note:
1. The current setting is to train two unets (one to estimate the residuals and one to estimate the noise), which can be used to explore partially path-independent generation process.
2. Other tasks need to modify a) `[self.alphas_cumsum[t]*self.num_timesteps, self.betas_cumsum[t]*self.num_timesteps]]` -> `[t,t]` (in [L852](https://github.com/nachifur/RDDM/blob/50d7dc3670a68dfe89c411a9445cc824b4fcd911/src/residual_denoising_diffusion_pytorch.py#L852) and [L1292](https://github.com/nachifur/RDDM/blob/50d7dc3670a68dfe89c411a9445cc824b4fcd911/src/residual_denoising_diffusion_pytorch.py#L1292)). b) For image restoration, `generation=False` in [L120](https://github.com/nachifur/RDDM/blob/ee4df22b672772a46b48251b0f56d82489d6adf0/train.py#L120), `convert_to_ddim=False` in [L640](https://github.com/nachifur/RDDM/blob/46ffd50f858a59fc3b43e538d501d991af3c1472/src/residual_denoising_diffusion_pytorch.py#L640)  and [L726](https://github.com/nachifur/RDDM/blob/11d06d5f389e3953fdedcf33fffbb81ff4d1583a/src/residual_denoising_diffusion_pytorch.py#L726C9-L726C31). c) **modify the corresponding experimental settings (see Table 4 in the Appendix)**.
3. The code is being updated.

## Requirements

To install requirements:

```
conda env create -f install.yaml
```

## Dataset

[Raindrop](https://github.com/rui1996/DeRaindrop)

[GoPro](https://github.com/swz30/MPRNet/blob/main/Deblurring/Datasets/README.md)

[ISTD](https://github.com/DeepInsight-PCALab/ST-CGAN)

SID-RGB: [kexu](https://kkbless.github.io/) or [download](https://drive.google.com/drive/folders/1-psXDjeW4FiRdLjc9idABsxGPo1Kn1jR)

[LOL](https://daooshee.github.io/BMVC2018website/)

[CelebA](https://github.com/nachifur/RDDM/issues/8#issuecomment-1978889073)

## Training

To train RDDM, run this command:

```train
python train.py
```
or
```train
accelerate launch train.py
```

## Evaluation

To evaluate image generation, run:

```eval
cd eval/image_generation_eval/
python fid_and_inception_score.py path_of_gen_img
```

For image restoration, MATLAB evaluation codes in `./eval`.

## Pre-trained Models

The pre-trained models will be provided later.


## Results

See Table 3 in main paper.

## Other experiments

We can convert a pre-trained DDIM to RDDM by coefficient transformation (see [code](https://github.com/nachifur/RDDM/tree/main/experiments/convert_pretrained_DDIM_to_RDDM)).

## Citation
If you find our work useful in your research, please consider citing:
```
@article{liu2023residual,
    title={Residual Denoising Diffusion Models}, 
    author={Jiawei Liu and Qiang Wang and Huijie Fan and Yinong Wang and Yandong Tang and Liangqiong Qu},
    year={2023},
    journal={arXiv preprint arxiv:2308.13712}
}
```
## Contact
Please contact Liangqiong Qu (https://liangqiong.github.io/) or Jiawei Liu (liujiawei18@mails.ucas.ac.cn) if there is any question.
