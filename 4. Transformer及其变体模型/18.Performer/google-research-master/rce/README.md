# Replacing Rewards with Examples: Example-Based Policy Search via Recursive Classification

<p align="center"> Benjamin Eysenbach, &nbsp; Sergey Levine, &nbsp; Ruslan Salakhutdinov</p>
<p align="center"> NeurIPS 2021 (oral)</p>
<p align="center">
   <a href="http://arxiv.org/abs/2103.12656">paper</a>, <a href="https://ben-eysenbach.github.io/rce/">website</a>
</p>

**tldr**: In many scenarios, the user is unable to describe the task in words or numbers, but can readily provide examples of what the world would look like if the task were solved. To address this problem of *example-based control*, we propose an off-policy algorithm, *recursive-classification of examples* (RCE). In contrast to prior approaches, out method does require expert demonstrations and never learns an explicit reward function. Nonetheless, this method retains many of the theoretical properties of reward-based learning, such as Bellman equations, where the standard reward function term is replaced by data. Experiments show that our approach outperforms these prior approaches on a range of simulated robotic manipulation tasks.


If you use this code, please consider adding the corresponding citation:

```
@inproceedings{
  eysenbach2021replacing,
  title={Replacing Rewards with Examples: Example-Based Policy Search via Recursive Classification},
  author={Benjamin Eysenbach and Sergey Levine and Ruslan Salakhutdinov},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021},
}
```

## Installation
These instructions were tested in Google Cloud Compute with Ubuntu version 18.04.




### 1. Install Mujoco
Copy your mujoco key to `~/.mujoco/mjkey.txt`, then complete the steps below:

```
sudo apt install unzip gcc libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
sudo ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so
wget https://www.roboti.us/download/mujoco200_linux.zip -P /tmp
unzip /tmp/mujoco200_linux.zip -d ~/.mujoco
mv ~/.mujoco/mujoco200_linux ~/.mujoco/mujoco200
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin" >> ~/.bashrc
```

### 2. Install Anaconda
```
wget https://repo.anaconda.com/miniconda/Miniconda2-latest-Linux-x86_64.sh
chmod +x Miniconda2-latest-Linux-x86_64.sh
./Miniconda2-latest-Linux-x86_64.sh
```
Restart your terminal so the changes take effect.


### 3. Create an Anaconda environment and install the remaining dependencies

*Updated 12/28/22. Thanks to Kevin Lin and Mingdong Wu for developing these revised instructions.*

* `conda env create -f environment.yml`
* `git clone https://github.com/rail-berkeley/d4rl.git`
* In `d4rl/setup.py`, remove installation requirements of `mjrl@master` and `dm_control@master`.
* In the d4rl repository, run `pip install -e .`


## Running Experiments

The following lines replicate the RCE experiments on the Sawyer tasks and Adept task using state-based observations. Please see the discussion below for an explanation of what the various command line options do.


```
python train_eval.py --root_dir=~/rce/sawyer_drawer_open --gin_bindings='train_eval.env_name="sawyer_drawer_open"'

python train_eval.py --root_dir=~/rce/sawyer_push --gin_bindings='train_eval.env_name="sawyer_push"'

python train_eval.py --root_dir=~/rce/sawyer_lift --gin_bindings='train_eval.env_name="sawyer_lift"'

python train_eval.py --root_dir=~/rce/door --gin_bindings='train_eval.env_name="door-human-v0"'

python train_eval.py --root_dir=~/rce/sawyer_box_close --gin_bindings='train_eval.env_name="sawyer_box_close"'

python train_eval.py --root_dir=~/rce/sawyer_bin_picking --gin_bindings='train_eval.env_name="sawyer_bin_picking"' --gin_bindings='critic_loss.q_combinator="max"' --gin_bindings='actor_loss.q_combinator="max"'

python train_eval.py --root_dir=~/rce/hammer --gin_bindings='train_eval.env_name="hammer-human-v0"'
```

To run SQIL, add the additional arguments:

```--gin_bindings='critic_loss.loss_name="q"' --gin_bindings='train_eval.n_step=None'```
 
### Questions?
If you have any questions, comments, or suggestions, please reach out to Benjamin Eysenbach (eysenbach@google.com).
