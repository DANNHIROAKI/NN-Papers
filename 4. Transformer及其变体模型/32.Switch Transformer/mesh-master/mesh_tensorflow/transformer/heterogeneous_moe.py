# coding=utf-8
# Copyright 2022 The Mesh TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Heterogeneous Mixture-of-experts code.

Includes the option to vary effective depth and width of experts
by masking out parts of experts

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import mesh_tensorflow as mtf
from mesh_tensorflow.transformer import moe
from mesh_tensorflow.transformer import transformer
import numpy as np
import tensorflow.compat.v1 as tf


@gin.configurable
class HeterogeneousMoE1D(transformer.TransformerLayer):
  """heterogeneous Mixture of Experts Layer."""

  def __init__(self,
               num_experts=16,
               loss_coef=1e-2,
               hidden_size=4096,
               group_size=1024,
               capacity_factor_train=1.25,
               capacity_factor_eval=2.0,
               use_second_place_loss=False,
               second_policy_train="random",
               second_policy_eval="random",
               second_threshold_train=0.2,
               second_threshold_eval=0.2,
               dropout_rate=0.0,
               activation="relu",
               moe_gating="top_2",
               min_expert_capacity=4,
               switch_policy_train="input_jitter",
               switch_policy_eval="input_jitter",
               switch_dropout=0.1,
               switch_temperature=1.0,
               switch_jitter=1e-2,
               ntlb_top_k=4,
               output_dim=None,
               z_loss=None,
               word_embed_mode=None,
               use_second_place_expert_prob=None,
               use_second_place_expert_prob_temp=None,
               moe_num_layers=1,
               heterogeneous_mask_info=None,
               top_n_num_experts_per_token=3):
    self._hparams = moe.HParams(
        moe_gating=moe_gating,
        moe_num_experts=num_experts,
        moe_loss_coef=loss_coef,
        moe_hidden_size=hidden_size,
        moe_group_size=group_size,
        moe_min_expert_capacity=min_expert_capacity,
        moe_capacity_factor_train=capacity_factor_train,
        moe_capacity_factor_eval=capacity_factor_eval,
        moe_use_second_place_loss=use_second_place_loss,
        moe_second_policy_train=second_policy_train,
        moe_second_policy_eval=second_policy_eval,
        moe_second_threshold_train=second_threshold_train,
        moe_second_threshold_eval=second_threshold_eval,
        moe_dropout_rate=dropout_rate,
        moe_switch_policy_train=switch_policy_train,
        moe_switch_policy_eval=switch_policy_eval,
        moe_switch_dropout=switch_dropout,
        moe_switch_temperature=switch_temperature,
        moe_switch_jitter=switch_jitter,
        moe_output_dim=output_dim,
        moe_ntlb_top_k=ntlb_top_k,
        moe_z_loss=z_loss,
        moe_word_embed_mode=word_embed_mode,
        moe_use_second_place_expert_prob=(
            use_second_place_expert_prob),
        moe_use_second_place_expert_prob_temp=(
            use_second_place_expert_prob_temp),
        moe_top_n_num_experts_per_token=top_n_num_experts_per_token,
        moe_num_layers=moe_num_layers,
        moe_heterogeneous_mask_info=heterogeneous_mask_info)
    self._activation = activation

  def call(self, context, x, losses=None):
    """Call the layer."""
    if context.model.ensemble_dim:
      raise NotImplementedError("MoE not yet implemented with ensembles")

    has_length_dim = context.length_dim in x.shape.dims
    if not has_length_dim:
      x_shape = x.shape
      shape_with_length = mtf.Shape(
          x_shape.dims[:-1] + [mtf.Dimension("length", 1)]
          + x_shape.dims[-1:])
      x = mtf.reshape(x, shape_with_length)

    # Extract the MoE output dimension
    if self._hparams.moe_output_dim is not None:
      output_dim = self._hparams.moe_output_dim
    else:
      output_dim = context.model.model_dim
    y, loss = transformer_moe_layer_v1(
        x,
        output_dim,
        self._hparams,
        context.train,
        context.variable_dtype,
        layout=context.model.layout,
        mesh_shape=context.model.mesh_shape,
        nonpadding=context.nonpadding,
        activation=self._activation,
        num_microbatches=context.num_microbatches,
        token_embeddings=context.input_embeddings,
        context=context)
    if context.losses is not None:
      context.losses.append(loss)
    if not has_length_dim:
      y = mtf.reshape(y, x_shape)
    return y


def transformer_moe_layer_v1(
    inputs, output_dim, hparams, train, variable_dtype,
    layout=None, mesh_shape=None, nonpadding=None, activation=mtf.relu,
    num_microbatches=None, token_embeddings=None, context=None):
  """Local heterogenous mixture of experts.

  See transformer_moe_layer_v1 in moe.py for a more detailed explanation for
  a generic moe layer.

  The heterogeneous mask outputted by generate_heterogeneous_expert_masks has
  dimension [maximum hidden size, maximum # layers, # experts] and its shape
  will overwrite the parameters moe_num_layers and moe_hidden_size in hparams.
  The layer-specific mask slice is applied at each expert layer to the
  activation which is [expert width, # experts]. If the heterogeneous_mask_info
  is None, there is no mask applied and the code is equivalent to the
  homogeneous case.


  The input is n-dimensional: [<batch_and_length_dims>, input_dim], consisting
  of the representations of all positions in a batch of sequences.

  Each position of each sequence is sent to 0-2 experts.  The expert
  choices and the combination weights are determined by a learned gating
  function.

  This function returns a small auxiliary loss that should be added to the
  training loss of the model.  This loss helps to balance expert usage.
  Without the loss, it is very likely that a few experts will be trained and
  the rest will starve.

  Dimensions cheat sheet:
  B: batch dim(s)
  L: original sequence length
  M: input depth
  N: output depth
  G: number of groups
  S: group size
  E: number of experts
  C: expert capacity

  Args:
    inputs: a mtf.Tensor with shape [batch_dim(s), length_dim, input_dim]
    output_dim: a mtf.Dimension (for Transformer, this is input_dim)
    hparams: model hyperparameters
    train: a boolean
    variable_dtype: a mtf.VariableDType
    layout: optional - an input to mtf.convert_to_layout_rules
    mesh_shape: optional - an input to mtf.convert_to_shape
    nonpadding: an optional Tensor with shape [batch_dim(s), length_dim]
      and the same dtype as inputs, consisting of ones(nonpadding)
      and zeros(padding).
    activation: a function.
    num_microbatches: number of microbatches.
    token_embeddings: a mtf.Tensor with shape
      [batch_dim(s), length_dim, input_dim]. These are the word embeddings for
      that correspond to the inputs. These can optionally be used to make
      routing decisions.
    context: a Context.

  Returns:
    outputs: a Tensor with shape [batch_dim(s), length_dim, output_dim]
    loss: a mtf scalar

  Raises:
    ValueError: on unrecognized hparams.moe_gating
  """
  orig_inputs = inputs

  experts_dim = mtf.Dimension("experts", hparams.moe_num_experts)

  if hparams.moe_heterogeneous_mask_info is not None:
    tf.logging.info("moe_heterogeneous_mask_info: {}".format(
        hparams.moe_heterogeneous_mask_info))
    heterogeneous_mask = generate_heterogeneous_expert_masks(
        hparams.moe_heterogeneous_mask_info,
        hparams.moe_num_experts,
        experts_dim,
        mesh=inputs.mesh,
        expert_width=hparams.moe_hidden_size)
    # overwrite depth and width with the mask maximum dimension
    hparams.moe_num_layers = heterogeneous_mask.shape[1].size
    hparams.moe_hidden_size = heterogeneous_mask.shape[0].size
  hidden_dim = mtf.Dimension("expert_hidden", hparams.moe_hidden_size)

  # We "cheat" here and look at the mesh shape and layout. This is to ensure
  # that the number of groups is a multiple of the mesh dimension
  # over which those groups are split.
  batch_and_length_dims, input_dim = (orig_inputs.shape.dims[:-1],
                                      orig_inputs.shape.dims[-1])
  # Hack: we assume that
  #   "outer_batch" == replication of experts
  #   mesh_dim_size can be derived from mesh_shape and orig_batch_dim
  #
  # We then reqire num_groups to be a multiple of mesh_dim_size.
  if orig_inputs.shape.dims[0].name == "outer_batch":
    outer_batch_dim, orig_batch_dim = orig_inputs.shape.dims[:2]
  else:
    outer_batch_dim, orig_batch_dim = (mtf.Dimension("outer_batch", 1),
                                       orig_inputs.shape.dims[0])

  # Number of MoE inputs (total number of position across batch_and_length_dims
  # per replica.
  n = 1
  for d in batch_and_length_dims:
    n *= d.size

  n = n // outer_batch_dim.size

  mesh_dim_size = mtf.tensor_dim_to_mesh_dim_size(layout, mesh_shape,
                                                  orig_batch_dim)
  num_groups, group_size = moe._split_into_groups(  # pylint: disable=protected-access
      n, hparams.moe_group_size, mesh_dim_size)
  # TODO(barretzoph): implementation without pylint calls?

  group_size_dim = mtf.Dimension("group", group_size)
  num_groups_dim = mtf.Dimension(orig_batch_dim.name, num_groups)

  moe_input_dims = [outer_batch_dim, num_groups_dim, group_size_dim, input_dim]
  # OGSM Tensor
  inputs = mtf.reshape(inputs, moe_input_dims)

  # Token embeddings that can be optionally used in the router for determining
  # where to send tokens.
  if hparams.moe_word_embed_mode is not None:
    token_embeddings = mtf.cast(
        mtf.reshape(token_embeddings, moe_input_dims), inputs.dtype)

  # Each sequence sends expert_capacity positions to each expert.
  if train:
    capacity_factor = hparams.moe_capacity_factor_train
  else:
    capacity_factor = hparams.moe_capacity_factor_eval
  expert_capacity = min(
      group_size_dim.size,
      int((group_size_dim.size * capacity_factor) / experts_dim.size))
  expert_capacity = max(expert_capacity, hparams.moe_min_expert_capacity)
  tf.logging.info("expert_capacity: %d" % expert_capacity)
  expert_capacity_dim = mtf.Dimension("expert_capacity", expert_capacity)
  experts_dim_unsplit = mtf.Dimension("expert_unsplit", experts_dim.size)
  batch_dim_unsplit = mtf.Dimension("batch_unsplit", num_groups_dim.size)
  if nonpadding is not None:
    nonpadding = mtf.zeros(
        inputs.mesh, batch_and_length_dims, dtype=inputs.dtype) + nonpadding
    nonpadding = mtf.reshape(nonpadding, moe_input_dims[:-1])
  if hparams.moe_gating == "top_2":
    # combine_tensor,
    # dispatch_tensor  OG`SEC Tensors
    # (G is generally split along mesh dim)
    dispatch_tensor, combine_tensor, loss = moe._top_2_gating(  # pylint: disable=protected-access
        inputs=inputs,
        outer_expert_dims=None,
        experts_dim=experts_dim_unsplit,
        expert_capacity_dim=expert_capacity_dim,
        hparams=hparams,
        train=train,
        variable_dtype=variable_dtype,
        importance=nonpadding,
        num_microbatches=num_microbatches,
        token_embeddings=token_embeddings)
  elif hparams.moe_gating == "top_n":
    dispatch_tensor, combine_tensor, loss = moe._top_n_gating(  # pylint: disable=protected-access
        inputs=inputs,
        outer_expert_dims=None,
        experts_dim=experts_dim_unsplit,
        expert_capacity_dim=expert_capacity_dim,
        hparams=hparams,
        train=train,
        variable_dtype=variable_dtype,
        importance=nonpadding,
        num_microbatches=num_microbatches,
        token_embeddings=token_embeddings)
  elif hparams.moe_gating == "switch":
    dispatch_tensor, combine_tensor, loss = moe._switch_gating(  # pylint: disable=protected-access
        inputs=inputs,
        outer_expert_dims=None,
        experts_dim=experts_dim_unsplit,
        expert_capacity_dim=expert_capacity_dim,
        hparams=hparams,
        train=train,
        variable_dtype=variable_dtype,
        importance=nonpadding,
        num_microbatches=num_microbatches,
        token_embeddings=token_embeddings)
  elif hparams.moe_gating == "ntlb":
    dispatch_tensor, combine_tensor, loss = moe._ntlb_gating(  # pylint: disable=protected-access
        inputs=inputs,
        outer_expert_dims=None,
        experts_dim=experts_dim_unsplit,
        expert_capacity_dim=expert_capacity_dim,
        hparams=hparams,
        train=train,
        variable_dtype=variable_dtype,
        importance=nonpadding,
        num_microbatches=num_microbatches,
        token_embeddings=token_embeddings)
  elif hparams.moe_gating == "switch_max":
    dispatch_tensor, combine_tensor, loss = moe._switch_max_gating(  # pylint: disable=protected-access
        inputs=inputs,
        outer_expert_dims=None,
        experts_dim=experts_dim_unsplit,
        expert_capacity_dim=expert_capacity_dim,
        hparams=hparams,
        train=train,
        variable_dtype=variable_dtype,
        importance=nonpadding,
        num_microbatches=num_microbatches,
        token_embeddings=token_embeddings)
  elif hparams.moe_gating == "expert_selection":
    dispatch_tensor, combine_tensor, loss = moe._expert_selection_gating(  # pylint: disable=protected-access
        inputs=inputs,
        outer_expert_dims=None,
        experts_dim=experts_dim_unsplit,
        group_size_dim=group_size_dim,
        expert_capacity_dim=expert_capacity_dim,
        hparams=hparams,
        train=train,
        variable_dtype=variable_dtype,
        importance=nonpadding,
        name="expert_selection_gating",
        num_microbatches=num_microbatches,
        token_embeddings=token_embeddings)
  else:
    raise ValueError("unknown hparams.moe_gating=%s" % hparams.moe_gating)

  expert_inputs = mtf.einsum([inputs, dispatch_tensor],
                             mtf.Shape([
                                 outer_batch_dim, experts_dim_unsplit,
                                 num_groups_dim, expert_capacity_dim, input_dim
                             ]))

  # Extra reshape reduces communication cost for model-parallel versions.
  # For model-parallel versions, this reshape causes an mtf.slice and for non-
  # model-parallel versions, this has no effect.
  d_model_split_dim = mtf.Dimension("d_model_split", input_dim.size)
  expert_inputs = mtf.reshape(
      expert_inputs,
      mtf.Shape([
          outer_batch_dim, experts_dim, batch_dim_unsplit, expert_capacity_dim,
          d_model_split_dim
      ]))

  # Split over batch -> split over experts
  expert_inputs = mtf.reshape(
      expert_inputs,
      mtf.Shape([
          outer_batch_dim, experts_dim, batch_dim_unsplit, expert_capacity_dim,
          input_dim
      ]))

  # Pretend we have heterogeneous_mask with shape [moe_num_layers, num_experts]
  for layer_idx in range(hparams.moe_num_layers):
    with tf.variable_scope("expert_layer_{}".format(layer_idx)):
      res_h = 0.0
      if layer_idx > 0:
        res_h = expert_inputs
        expert_inputs = transformer.sublayer_rms_norm(
            expert_inputs, None, context)

      # Now feed the expert inputs through the experts.
      h = mtf.layers.dense_product(
          expert_inputs,
          reduced_dims=expert_inputs.shape.dims[-1:],
          new_dims=[hidden_dim],
          expert_dims=[experts_dim],
          activation_functions=activation, use_bias=False,
          variable_dtype=variable_dtype, name="wi")

      # apply dropout
      if hparams.moe_dropout_rate != 0.0:
        h = mtf.dropout(h, is_training=train,
                        keep_prob=1.0 - hparams.moe_dropout_rate)
      # only if heterogeneous
      if hparams.moe_heterogeneous_mask_info is not None:
        # Get mask for current layer by slicing heterogeneous mask
        heterogeneous_mask_slice = mtf.slice(
            heterogeneous_mask, layer_idx, 1, "num_expert_layers")

        # Get rid of the expert layers dimension.
        heterogeneous_mask_slice = mtf.reshape(
            heterogeneous_mask_slice,
            [heterogeneous_mask_slice.shape[0],
             heterogeneous_mask_slice.shape[-1]])
        h *= mtf.cast(heterogeneous_mask_slice, h.dtype)
      expert_output = mtf.layers.dense(
          h, output_dim, expert_dims=[experts_dim], use_bias=False,
          reduced_dims=h.shape.dims[-1:], variable_dtype=variable_dtype,
          name="wo")

      if layer_idx < (hparams.moe_num_layers - 1):
        expert_output = transformer.sublayer_dropout(
            expert_output, None, context)
      expert_output += res_h
      expert_inputs = expert_output

  # Extra reshape reduces communication cost for model-parallel versions.
  # For model-parallel versions, this reshape causes an mtf.slice and for non-
  # model-parallel versions, this has no effect.
  expert_output = mtf.reshape(
      expert_output,
      mtf.Shape([
          outer_batch_dim, experts_dim_unsplit, num_groups_dim,
          expert_capacity_dim, d_model_split_dim
      ]))

  # Split over experts -> split over batch
  expert_output = mtf.reshape(
      expert_output,
      mtf.Shape([
          outer_batch_dim,
          experts_dim_unsplit,
          num_groups_dim,
          expert_capacity_dim,
          output_dim,
      ]))
  moe_output_dims = moe_input_dims[:-1] + [output_dim]
  output = mtf.einsum([expert_output, combine_tensor],
                      mtf.Shape(moe_output_dims))
  output = mtf.reshape(output, batch_and_length_dims + [output_dim])
  return output, loss * hparams.moe_loss_coef


def generate_heterogeneous_expert_masks(
    mask_info, num_experts, experts_dim, mesh, expert_width):
  """Generates the heterogeous expert masks.

  Example mask_info format:
    mask_info = [{'percent_number': .5, 'layers': 1, 'width':1},
                 {'percent_number': .5, 'layers': 2, 'width':2}]

  Args:
    mask_info: list of dicts.
    num_experts: number of experts in the model
    experts_dim: mtf dimension for experts (partitioned)
    mesh: mesh object
    expert_width: int, default expert width which will be modified by the mask

  Returns:
    mask of shape [moe_num_layers, num_experts, hidden_size].
  """
  # Get max num layers
  max_layers = max([m["layers"] for m in mask_info])
  # Get max width
  max_width = max([m["width"] for m in mask_info])
  # Will be shape [max_width, max_layers, num_experts]
  expert_mask = np.zeros([max_width, max_layers, 0])
  for idx, mask_i in enumerate(mask_info):
    if mask_i["percent_number"] < 1.0:
      num_experts_in_mask = int(num_experts * mask_i["percent_number"])
    else:
      num_experts_in_mask = int(mask_i["percent_number"])
    # if percent_number=1 either because homogeneous experts or just 1 expert
    # in which case num_experts_in_mask will be reset to num_experts
    # creating one homogeneous group
    if idx == (len(mask_info) - 1):  # Last position
      num_experts_in_mask_tmp = num_experts - expert_mask.shape[2]
      if num_experts_in_mask_tmp != num_experts_in_mask:
        tf.logging.info(
            "Expert layer probabilities do not evenly divide "
            "the number of experts: {} {}".format(
                num_experts_in_mask, num_experts_in_mask_tmp))
        num_experts_in_mask = num_experts_in_mask_tmp
    mask = np.zeros([int(max_width), int(max_layers),
                     num_experts_in_mask])
    # Zero out the last layers of the experts.
    mask[:(mask_i["width"]*expert_width), :mask_i["layers"], :] = 1
    expert_mask = np.concatenate([expert_mask, mask], axis=2)  # expert dim
  assert expert_mask.shape[2] == num_experts
  tf.logging.info("heterogeneous mask: {}".format(expert_mask))

  # Now import the numpy mask into Mesh TF.
  layers_dim = mtf.Dimension("num_expert_layers", max_layers)
  width_dim = mtf.Dimension("expert_hidden", max_width)
  expert_mask_tf = tf.convert_to_tensor(expert_mask)
  expert_mask_mtf = mtf.import_tf_tensor(
      mesh, tf_tensor=expert_mask_tf,
      shape=[width_dim, layers_dim, experts_dim])
  return expert_mask_mtf
