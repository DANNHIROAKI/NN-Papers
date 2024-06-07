This is the codebase for [Observation-Guided Diffusion Probabilistic Models](https://arxiv.org/abs/2310.04041v1). This repository is based on [NVlabs/edm](https://github.com/NVlabs/edm).
The repository for ADM baseline can be found at [Junoh-Kang/OGDM_adm](https://github.com/Junoh-Kang/OGDM_adm).

# Dependencies
We share the environment of the code by docker.
```
docker pull snucvlab/ogdm:edm
```

# Model training (fine-tune)
```
torchrun --standalone --nproc_per_node=[N] train.py --outdir=[outdir] --data=[data_dir] --cond=0 --arch=ddpmpp --disc=True --gamma=[gamma] --k=[k] --transfer=[model_to_finetune] --duration=[duration] --batch-gpu=[B] --ddim=[True/False] 

# Euler projection
torchrun --standalone --nproc_per_node=4 train.py --outdir=logs --data=data/cifar10_32 --cond=0 --arch=ddpmpp --disc=True --k=0.2 --gamma=0.025 --transfer=logs/baseline/network-snapshot-200000.pkl --duration=20 --batch-gpu=64 --ddim=True

# Huen's projection
torchrun --standalone --nproc_per_node=4 train.py --outdir=logs --data=data/cifar10_32 --cond=0 --arch=ddpmpp --disc=True --k=0.2 --gamma=0.005 --transfer=logs/baseline/network-snapshot-200000.pkl --duration=20 --batch-gpu=64 --ddim=False
```

# Model Sampling
```
torchrun --standalone --nproc_per_node=[N] generate.py --seeds=[0-n] --batch=[B] --network=[model_to_sample] --disc=edm --schedule=linear --scaling=none --solver=[euler/pndm/edm] --steps=[steps]

# For NFE=15
# Euler method (NFE = steps)
torchrun --standalone --nproc_per_node=4 generate.py --seeds=0-49999 --batch=512 --network=logs/euler/network-snapshot-020000.pkl --disc=edm --schedule=linear --scaling=none --solver=euler --steps=15

# S-PNDM (NFE = steps + 1)
torchrun --standalone --nproc_per_node=4 generate.py --seeds=0-49999 --batch=512 --network=logs/euler/network-snapshot-020000.pkl --solver=pndm --steps=14

# Heun's method (NFE = 2*steps-1)
torchrun --standalone --nproc_per_node=4 generate.py --seeds=0-49999 --batch=512 --network=logs/heun/network-snapshot-020000.pkl --steps=8
```

# Evaluation
We share the environment of the evaluation code by docker.
```
docker pull snucvlab/ogdm:fid
```

Then, run
```
# When activations of reference is not ready, evaluate and save activations at the same time by
python fid_prdc.py [reference] [sample] -save_act_path [ref.npz]

# When activations of reference is ready, evaluate 
python fid_prdc.py [reference] [sample]
```
to evaluate.

# Download trained models
We provide trained models for CIFAR-10.
The downloaded files should look like this:
```
└── project
    └── network-snapshot-xxxxxx.pkl 
```
Here are links to download and wget commands.
- CIFAR-10 baseline: [cifar10_baseline_edm.tar.gz](https://drive.google.com/file/d/122I_liApvIKN3cMuzBAMt9IpW0bNrAKd/view?usp=share_link)
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=122I_liApvIKN3cMuzBAMt9IpW0bNrAKd' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1F7deiE3_hAITp-G74B4s61PWyWnKcjT5" -O cifar10_baseline_edm.tar.gz && rm -rf /tmp/cookies.txt
```

- CIFAR-10 Euler projection: [cifar10_euler_edm.tar.gz](https://drive.google.com/file/d/1YTPih5_Spjw2lmPrU9JBfCKRFwmLk3rs/view?usp=share_link)
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1YTPih5_Spjw2lmPrU9JBfCKRFwmLk3rs' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1F7deiE3_hAITp-G74B4s61PWyWnKcjT5" -O cifar10_euler_edm.tar.gz && rm -rf /tmp/cookies.txt
```

- CIFAR-10 Heun's projection: [cifar10_heun_edm.tar.gz](https://drive.google.com/file/d/1v2fv_6DqaZfq4Vl7kqqHOUDRjwraYbHl/view?usp=share_link)
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1v2fv_6DqaZfq4Vl7kqqHOUDRjwraYbHl' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1F7deiE3_hAITp-G74B4s61PWyWnKcjT5" -O cifar10_heun_edm.tar.gz && rm -rf /tmp/cookies.txt
```

# Citation
```
@inproceedings{kang2023odgm,
  author    = {Junoh Kang and Jinyoung Choi and Sungik Choi and Bohyung Han},
  title     = {Observation-Guided Diffusion Probabilistic Models},
  booktitle = {},
  year      = {2023}
}
```

# Acknowledgments

