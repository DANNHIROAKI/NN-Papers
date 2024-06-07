# MSG with Batch Ensembles
Implements MSG using Batch Ensembles. If seeking to use or implement alternative
forms of MSG (e.g. using different ensembling techniques than the default deep
ensembles used in default MSG), please refer to the note in 
`jrl/agents/msg/README.md`.

Please remember that some flags are "global", meaning that they are defined in
`runner_flags.py`, while algorithm specific parameters are configured using
gin configs.

All agents have a config parameter named `num_sgd_steps_per_step`, which
determines how many training steps we perform per call to the learner's step
function. Setting this to a larger number allows Jax to perform optimizations
that make training faster. Keep in mind that you should set the `batch_size`
parameter to `(num_sgd_steps_per_step) x (per training step batch size that you want)`.
Also, you should set `num_steps` to
`(total number of training steps you want) / num_sgd_steps_per_step`.
`batch_ensemble_msg.config.BatchEnsembleMSGConfig.num_bc_iters` and `batch_ensemble_msg.config.BatchEnsembleMSGConfig.pretrain_iters`
are set in terms of "true" number of steps, i.e. no need to account for
`num_sgd_steps_per_step`.
For easier local debugging, you can set:
```
--num_sgd_steps_per_step 1 \
--batch_size 64 \
--num_steps 1000 \
--episodes_per_eval 10 \
--gin_bindings='batch_ensemble_msg.config.BatchEnsembleMSGConfig.num_sgd_steps_per_step=1'
```

Note: Only for `halfcheetah, hopper, walker` experiments we set
`batch_ensemble_msg.config.BatchEnsembleMSGConfig.use_double_q=True`. It was not a noticeable difference
but we did not go back to rerun full experiments with this param set to True.

Note: For additional parameters that can be set please refer to `batch_ensemble_msg/config.py`

## Running MSG with Batch Ensembles
```
python3 -m jrl.localized.runner \
--pdb_post_mortem \
--debug_nans=False \
--create_saved_model_actor=False \
--num_steps 11000 \
--eval_every_steps 500 \
--episodes_per_eval 100 \
--batch_size 51200 \
--root_dir '/tmp/test_msg_deep_ensembles' \
--seed 42 \
--algorithm 'batch_ensemble_msg' \
--task_class 'd4rl' \
--task_name 'antmaze-large-diverse-v0' \
--gin_bindings='batch_ensemble_msg.config.BatchEnsembleMSGConfig.num_sgd_steps_per_step=200' \
--gin_bindings='batch_ensemble_msg.config.BatchEnsembleMSGConfig.ensemble_size=64' \
--gin_bindings='batch_ensemble_msg.config.BatchEnsembleMSGConfig.beta=-8' \
--gin_bindings='batch_ensemble_msg.config.BatchEnsembleMSGConfig.behavior_regularization_alpha=0.1' \
--gin_bindings='batch_ensemble_msg.config.BatchEnsembleMSGConfig.behavior_regularization_type="v1"' \
--gin_bindings='batch_ensemble_msg.config.BatchEnsembleMSGConfig.actor_hidden_sizes=(256, 256, 256)' \
--gin_bindings='batch_ensemble_msg.config.BatchEnsembleMSGConfig.q_hidden_sizes=(256, 256, 256)' \
--gin_bindings='batch_ensemble_msg.config.BatchEnsembleMSGConfig.use_double_q=False'
```
