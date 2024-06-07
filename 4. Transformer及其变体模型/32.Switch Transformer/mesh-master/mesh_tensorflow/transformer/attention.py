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

"""Implementation of various types of attention."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mesh_tensorflow as mtf

import tensorflow.compat.v1 as tf


def attention(q,
              k,
              v,
              memory_length_dim,
              key_dim,
              value_dim,
              bias=None,
              dropout_rate=0.0,
              dropout_broadcast_dims=None,
              extra_logit=None,
              context=None,
              float32_logits=True,
              z_loss_coeff=None):
  """Dot-product attention - doesn't use positional dimensions.

  key_dim is a Dimension representing the channels in the queries and keys
  value_dim is a Dimension representing the channels in values
  memory_length_dim is a Dimension representing the different key/value pairs.

  Dimensions of q: other_query_dims + {key_dim}
  Dimensions of k: other_memory_dims + {memory_length_dim, key_dim}
  Dimensions of v: other_memory_dims + {memory_length_dim, value_dim}
  other_memory_dims is a subset of other_query_dims

  Typically, other_query_dims={batch, heads, length}
  Typically, other_memory_dims={batch, heads}

  Args:
    q: a Tensor
    k: a Tensor
    v: a Tensor
    memory_length_dim: a Dimension
    key_dim: a Dimension
    value_dim: a Dimension
    bias: a Tensor to be added into the attention logits.
    dropout_rate: a float.
    dropout_broadcast_dims: an optional list of mtf.Dimension
    extra_logit: an optional scalar or tensor
    context: an optional Transformer.Context
    float32_logits: a boolean - if True, then compute logits in float32 to avoid
      numerical issues with bfloat16
    z_loss_coeff: a float, if z_loss_coeff is not None then add an auxiliary
      loss to push the attention logits closer to zero. This helps to stabilize
      model training.

  Returns:
    Tensor with shape q.shape - key_dim + value_dim
  """
  orig_q_shape = q.shape
  q, k, v, bias = maybe_reshape_attention_input_for_2d_sharding(
      context, q, k, v, bias, [key_dim, value_dim])
  if float32_logits:
    k = mtf.cast(k, tf.float32)
    q = mtf.cast(q, tf.float32)
  logits = mtf.layers.us_einsum([q, k], reduced_dims=[key_dim])
  if bias is not None:
    logits += mtf.cast(bias, logits.dtype)

  # Adds auxiliary z-loss to push the attention logits towards zero.
  if z_loss_coeff is not None and context.train:
    tf.logging.info("attention z_loss being added: {}".format(
        tf.get_variable_scope().name))
    log_z = mtf.reduce_logsumexp(logits, memory_length_dim)
    z_loss = mtf.square(log_z) * mtf.cast(context.nonpadding, log_z.dtype)
    z_loss = mtf.reduce_mean(z_loss)
    if context.num_microbatches and context.num_microbatches > 1:
      tf.logging.info(
          "Dividing attention z-loss loss by num_microbatches={}".format(
              context.num_microbatches))
      z_loss /= context.num_microbatches
    if context.train:
      mtf.scalar_summary("attention_z_loss", z_loss)
    z_loss *= z_loss_coeff
    context.losses.append(mtf.cast(z_loss, v.dtype))

  weights = mtf.softmax(logits, memory_length_dim, extra_logit=extra_logit)
  weights = mtf.cast(weights, v.dtype)
  weights = mtf.dropout(
      weights, context.train, 1.0 - dropout_rate,
      noise_shape=weights.shape - dropout_broadcast_dims)
  outputs_shape = q.shape - key_dim + value_dim
  outputs = mtf.einsum([weights, v], outputs_shape)
  outputs = mtf.reshape(outputs, orig_q_shape - key_dim + value_dim)
  return outputs


def hybrid_attention(q,
                     k,
                     v,
                     context,
                     memory_length_dim,
                     key_dim,
                     value_dim,
                     bias=None,
                     dropout_rate=0.0,
                     dropout_broadcast_dims=None,
                     extra_logit=None):
  """Dot-product attention - doesn't use positional dimensions.

  key_dim is a Dimension representing the channels in the queries and keys
  value_dim is a Dimension representing the channels in values
  memory_length_dim is a Dimension representing the different key/value pairs.

  Dimensions of q: other_query_dims + {key_dim}
  Dimensions of k: other_memory_dims + {memory_length_dim, key_dim}
  Dimensions of v: other_memory_dims + {memory_length_dim, value_dim}
  other_memory_dims is a subset of other_query_dims

  Typically, other_query_dims={batch, heads, length}
  Typically, other_memory_dims={batch, heads}

  Args:
    q: a Tensor
    k: a Tensor
    v: a Tensor
    context: context of the attention layer.
    memory_length_dim: a Dimension
    key_dim: a Dimension
    value_dim: a Dimension
    bias: a Tensor to be added into the attention logits.
    dropout_rate: a float.
    dropout_broadcast_dims: an optional list of mtf.Dimension
    extra_logit: an optional scalar or tensor

  Returns:
    Tensor with shape q.shape - key_dim + value_dim
  """
  logits = mtf.layers.us_einsum([q, k], reduced_dims=[key_dim])
  if bias is not None:
    logits += bias

  query_length_dim = mtf.Dimension("length", memory_length_dim.size)
  doubly_coeff = mtf.get_variable(
      context.mesh, "doubly_coeff", [],
      initializer=tf.constant_initializer(0.5),
      dtype=context.variable_dtype)
  doubly_coeff = mtf.maximum(mtf.minimum(doubly_coeff, 1.), 0.)

  upper_weights = mtf.softmax(
      logits, memory_length_dim, extra_logit=extra_logit)

  lower_log_weights = mtf.log_softmax(
      logits, query_length_dim, extra_logit=extra_logit)
  doubly_weights = mtf.softmax(
      lower_log_weights, memory_length_dim, extra_logit=extra_logit)

  weights = doubly_coeff * doubly_weights + (1. - doubly_coeff) * upper_weights
  weights = mtf.dropout(
      weights, context.train, 1.0 - dropout_rate,
      noise_shape=weights.shape - dropout_broadcast_dims)
  outputs_shape = q.shape - key_dim + value_dim
  outputs = mtf.einsum([weights, v], outputs_shape)
  return outputs


def synthetic_attention(q,
                        k,
                        v,
                        memory_length_dim,
                        key_dim,
                        value_dim,
                        bias=None,
                        dropout_rate=0.0,
                        dropout_broadcast_dims=None,
                        extra_logit=None,
                        synthesize=True,
                        synthesize_mode="random_plus_alpha",
                        factorized_dim=16,
                        max_length=512,
                        context=None):
  """Synthetic Attention from Synthesizers (https://arxiv.org/abs/2005.00743).

  key_dim is a Dimension representing the channels in the queries and keys
  value_dim is a Dimension representing the channels in values
  memory_length_dim is a Dimension representing the different key/value pairs.

  Dimensions of q: other_query_dims + {key_dim}
  Dimensions of k: other_memory_dims + {memory_length_dim, key_dim}
  Dimensions of v: other_memory_dims + {memory_length_dim, value_dim}
  other_memory_dims is a subset of other_query_dims

  Typically, other_query_dims={batch, heads, length}
  Typically, other_memory_dims={batch, heads}

  Args:
    q: a Tensor
    k: a Tensor
    v: a Tensor
    memory_length_dim: a Dimension
    key_dim: a Dimension
    value_dim: a Dimension
    bias: a Tensor to be added into the attention logits.
    dropout_rate: a float.
    dropout_broadcast_dims: an optional list of mtf.Dimension
    extra_logit: an optional scalar or tensor
    synthesize: flag to use synthetic attention or not
    synthesize_mode: which variant of synthesizer to use
    factorized_dim: factorized dim for synthesizers
    max_length: max length of input sequence
    context: context since we need context mode

  Returns:
    Tensor with shape q.shape - key_dim + value_dim
  """

  if synthesize:
    num_heads = v.shape.get_dim_by_name("heads")
    tf.logging.info("Using synthesizer")
    if synthesize_mode == "random":
      tf.logging.info("Using Random Synthesizers")
      r_shape = mtf.Shape([mtf.Dimension("length", max_length),
                           mtf.Dimension("heads", num_heads.size),
                           mtf.Dimension("memory_length", max_length)])
      r = mtf.get_variable(context.mesh, "R", r_shape,
                           initializer=None,
                           dtype=context.variable_dtype)
      r = mtf.slice(r, 0, memory_length_dim.size, memory_length_dim.name)
      if context.mode == "incremental":
        r = mtf.gather(r, context.position, r.shape.get_dim_by_name("length"))
      else:
        length_dim = q.shape.get_dim_by_name("length")
        r = mtf.slice(r, 0, length_dim.size, "length")
      logits = r
      r_shape = logits.shape
    elif synthesize_mode == "factorized":
      tf.logging.info("Using Factorized Random Synthesizers")
      k = factorized_dim
      r1_shape = mtf.Shape([mtf.Dimension("tmp", k),
                            mtf.Dimension("heads", num_heads.size),
                            mtf.Dimension("memory_length", 512)])
      r2_shape = mtf.Shape([mtf.Dimension("tmp", k),
                            mtf.Dimension("heads", num_heads.size),
                            mtf.Dimension("memory_length", 512)])
      r_shape = mtf.Shape([mtf.Dimension("length", 512),
                           mtf.Dimension("heads", num_heads.size),
                           mtf.Dimension("memory_length", 512)])
      r1 = mtf.get_variable(context.mesh, "R1", r1_shape,
                            initializer=None,
                            dtype=context.variable_dtype)
      r2 = mtf.get_variable(context.mesh, "R2", r2_shape,
                            initializer=None,
                            dtype=context.variable_dtype)
      r = mtf.einsum([r1, r2], r_shape)
      r = mtf.slice(r, 0, memory_length_dim.size, memory_length_dim.name)
      if context.mode == "incremental":
        r = mtf.gather(r, context.position, r.shape.get_dim_by_name("length"))
      else:
        length_dim = q.shape.get_dim_by_name("length")
        r = mtf.slice(r, 0, length_dim.size, "length")
      logits = r
    elif synthesize_mode == "dense_minus":
      # Dense Synthesizer Model
      tmp_dim = mtf.Dimension("memory_length", max_length)
      logits = mtf.layers.dense(mtf.relu(q), [tmp_dim],
                                use_bias=False,
                                name="pi",
                                reduced_dims=[key_dim],
                                variable_dtype=None)
      logits = mtf.slice(logits, 0, memory_length_dim.size,
                         memory_length_dim.name)
      if context.mode == "incremental":
        pass
      else:
        length_dim = q.shape.get_dim_by_name("length")
        logits = mtf.slice(logits, 0, length_dim.size, "length")
    elif synthesize_mode == "random_plus_alpha" or \
        synthesize_mode == "random_plus":
      # Mixture Random Synthesizer with learnable Alpha
      tf.logging.info("Using Random Plus Alpha")
      logits = mtf.einsum([q, k], reduced_dims=[key_dim])
      num_heads = logits.shape.get_dim_by_name("heads")
      r_shape = mtf.Shape([mtf.Dimension("length", 512),
                           mtf.Dimension("heads", num_heads.size),
                           mtf.Dimension("memory_length", 512)])
      r = mtf.get_variable(context.mesh, "R", r_shape,
                           initializer=None,
                           dtype=context.variable_dtype)
      r = mtf.slice(r, 0, memory_length_dim.size, memory_length_dim.name)
      if context.mode == "incremental":
        r = mtf.gather(r, context.position, r.shape.get_dim_by_name("length"))
      else:
        length_dim = q.shape.get_dim_by_name("length")
        r = mtf.slice(r, 0, length_dim.size, length_dim.name)
      if "alpha" in synthesize_mode:
        alpha = mtf.get_variable(context.mesh,
                                 "alpha",
                                 mtf.Shape([mtf.Dimension("alpha", 1)]),
                                 initializer=tf.zeros_initializer(),
                                 dtype=context.variable_dtype)
        alpha = mtf.sigmoid(alpha)
        logits = ((1-alpha) * logits) + (alpha * r)
      else:
        logits = logits + r
    elif synthesize_mode == "dense_plus_alpha" or \
        synthesize_mode == "dense_plus":
      # Mixture Dense Synthesizer with learnable alpha
      tf.logging.info("Using Dense Plus Alpha Scaling")
      logits = mtf.einsum([q, k], reduced_dims=[key_dim])
      tmp_dim = mtf.Dimension("memory_length", 512)
      r = mtf.layers.dense(mtf.relu(q), [tmp_dim],
                           use_bias=False,
                           name="pi",
                           reduced_dims=[key_dim],
                           variable_dtype=None)
      r = mtf.slice(r, 0, memory_length_dim.size, memory_length_dim.name)
      if context.mode == "incremental":
        pass
      else:
        length_dim = q.shape.get_dim_by_name("length")
        r = mtf.slice(r, 0, length_dim.size, "length")
      if "alpha" in synthesize_mode:
        alpha = mtf.get_variable(context.mesh,
                                 "alpha",
                                 mtf.Shape([mtf.Dimension("alpha", 1)]),
                                 initializer=tf.zeros_initializer(),
                                 dtype=context.variable_dtype)
        alpha = mtf.sigmoid(alpha)
        logits = ((1-alpha) * logits) + (alpha * r)
      else:
        logits = logits + r
  if bias is not None:
    logits += bias

  weights = mtf.softmax(logits, memory_length_dim, extra_logit=extra_logit)
  weights = mtf.dropout(
      weights, context.train, 1.0 - dropout_rate,
      noise_shape=weights.shape - dropout_broadcast_dims)

  if synthesize and "plus" not in synthesize_mode:
    if synthesize_mode == "dense_minus":
      outputs_shape = mtf.Shape(q.shape.dims[:-1] + [value_dim])
    else:
      outputs_shape = mtf.Shape(q.shape.dims[:-1] + [num_heads, value_dim])
  else:
    outputs_shape = q.shape - [key_dim] + value_dim

  outputs = mtf.einsum([weights, v], outputs_shape)
  return outputs


class AttentionParams(object):
  """A set of parameters used for (multihead) attention."""

  def __init__(self,
               mesh,
               query_input_dim,
               memory_input_dim,
               output_dim,
               key_dim,
               value_dim,
               query_heads_dims,
               memory_heads_dims,
               variable_dtype,
               shared_kv=False,
               no_query=False,
               combine_dims=True,
               ensemble_dim=None,
               keep_query_heads_dims=False,
               fold_scaling_into_initializer=True,
               make_attention_vars=True):
    """Create attention parameters.

    combine_dims is a hack for faster execution.  The heads and key/value
    dimensions are combined in the variables and the computation.  The hack
    would not be necessary if XLA optimized einsum properly.

    Args:
      mesh: a Mesh
      query_input_dim: a Dimension
      memory_input_dim: a Dimension
      output_dim: a Dimension
      key_dim: a Dimension
      value_dim: a Dimension
      query_heads_dims: a list of Dimension
      memory_heads_dims: a list of Dimension
      variable_dtype: a mtf.VariableDType
      shared_kv: a boolean
      no_query: a boolean
      combine_dims: a boolean
      ensemble_dim: an optional Dimension
      keep_query_heads_dims: a boolean, if true keep the query_heads_dims in the
        output.
      fold_scaling_into_initializer: a boolean
      make_attention_vars: a boolean, whether to make the attention variables.
        This is typically True. Only set to False for ExpertsAttention which
        creates variables inside the moe.MoE1D-call.
    """
    if shared_kv and key_dim != value_dim:
      raise ValueError("shared_kv requires key_dim == value_dim")
    self.mesh = mesh
    self.query_input_dim = query_input_dim
    self.memory_input_dim = memory_input_dim
    self.output_dim = output_dim
    self.key_dim = key_dim
    self.value_dim = value_dim
    self.query_heads_dims = query_heads_dims or []
    self.memory_heads_dims = memory_heads_dims or []
    self.variable_dtype = variable_dtype
    self.shared_kv = shared_kv
    self.no_query = no_query
    self.combine_dims = combine_dims
    self.keep_query_heads_dims = keep_query_heads_dims
    self.fold_scaling_into_initializer = fold_scaling_into_initializer
    self.make_attention_vars = make_attention_vars

    if combine_dims:
      self.q_shape = [query_input_dim, _combined_dim(self.q_dims)]
      self.k_shape = [memory_input_dim, _combined_dim(self.k_dims)]
      self.v_shape = [memory_input_dim, _combined_dim(self.v_dims)]
      self.o_shape = [_combined_dim(self.o_dims), output_dim]
    else:
      self.q_shape = [query_input_dim] + self.q_dims
      self.k_shape = [memory_input_dim] + self.k_dims
      self.v_shape = [memory_input_dim] + self.v_dims
      self.o_shape = self.o_dims + [output_dim]
    if ensemble_dim:
      self.q_shape = [ensemble_dim] + self.q_shape
      self.k_shape = [ensemble_dim] + self.k_shape
      self.v_shape = [ensemble_dim] + self.v_shape
      self.o_shape = [ensemble_dim] + self.o_shape

    self.init_weights()

  def init_weights(self):
    """Initialize attention projection matrices."""
    if mtf.layers.unit_scaling_convention():
      init = tf.random_normal_initializer(stddev=1.0)
      q_init = init
      kv_init = init
      o_init = init
    else:
      stddev = self.query_input_dim.size ** -0.5
      if self.fold_scaling_into_initializer:
        stddev *= self.key_dim.size ** -0.5
      q_init = tf.random_normal_initializer(stddev=stddev)
      kv_init = tf.random_normal_initializer(
          stddev=self.memory_input_dim.size ** -0.5)
      o_init = tf.random_normal_initializer(
          stddev=mtf.Shape(self.query_heads_dims + [self.value_dim]).size**-0.5)

    # Toggle producing wq, wv, wk which are not needed for the ExpertsAttention
    if self.make_attention_vars:
      if not self.no_query:
        self.wq = mtf.get_variable(
            self.mesh,
            "q",
            self.q_shape,
            initializer=q_init,
            dtype=self.variable_dtype)
      if self.shared_kv:
        self.wkv = mtf.get_variable(
            self.mesh,
            "kv",
            self.k_shape,
            initializer=kv_init,
            dtype=self.variable_dtype)
      else:
        self.wk = mtf.get_variable(
            self.mesh,
            "k",
            self.k_shape,
            initializer=kv_init,
            dtype=self.variable_dtype)
        self.wv = mtf.get_variable(
            self.mesh,
            "v",
            self.v_shape,
            initializer=kv_init,
            dtype=self.variable_dtype)
    self.wo = mtf.get_variable(
        self.mesh,
        "o",
        self.o_shape,
        initializer=o_init,
        dtype=self.variable_dtype)

  def compute_q(self, query_antecedent):
    """Compute query Tensor q.

    Args:
      query_antecedent: a Tensor with dimensions
         {query_input_dim} + other_dims
    Returns:
      a Tensor with dimensions
         query_heads_dims + {key_dim} + other_dims
    """
    ret = mtf.layers.us_einsum(
        [query_antecedent, self.wq], reduced_dims=[self.query_input_dim])
    if self.combine_dims:
      ret = mtf.replace_dimensions(ret, ret.shape.dims[-1], self.q_dims)
    if not self.fold_scaling_into_initializer:
      ret *= self.key_dim.size ** -0.5
    return ret

  def compute_kv(self, memory_antecedent):
    """Compute key/value Tensor kv.

    Args:
      memory_antecedent: a Tensor with dimensions
        {memory_input_dim} + other_dims
    Returns:
      a Tensor with dimensions
        memory_heads_dims + {key_dim} + other_dims
    """
    if not self.shared_kv:
      raise ValueError("compute_kv can only be called with shared_kv")
    ret = mtf.layers.us_einsum(
        [memory_antecedent, self.wkv], reduced_dims=[self.memory_input_dim])
    if self.combine_dims:
      ret = mtf.replace_dimensions(ret, ret.shape.dims[-1], self.k_dims)
    return ret

  def compute_k(self, memory_antecedent):
    """Compute key Tensor k.

    Args:
      memory_antecedent: a Tensor with dimensions
        {memory_input_dim} + other_dims
    Returns:
      a Tensor with dimensions
        memory_heads_dims + {key_dim} + other_dims
    """
    if self.shared_kv:
      raise ValueError("compute_k cannot be called with shared_kv")
    ret = mtf.layers.us_einsum(
        [memory_antecedent, self.wk], reduced_dims=[self.memory_input_dim])
    if self.combine_dims:
      ret = mtf.replace_dimensions(ret, ret.shape.dims[-1], self.k_dims)
    return ret

  def compute_v(self, memory_antecedent):
    """Compute value Tensor v.

    Args:
      memory_antecedent: a Tensor with dimensions
        {memory_input_dim} + other_dims
    Returns:
      a Tensor with dimensions
        memory_heads_dims + {value_dim} + other_dims
    """
    if self.shared_kv:
      raise ValueError("compute_v cannot be called with shared_kv")
    ret = mtf.layers.us_einsum(
        [memory_antecedent, self.wv], reduced_dims=[self.memory_input_dim])
    if self.combine_dims:
      ret = mtf.replace_dimensions(ret, ret.shape.dims[-1], self.v_dims)
    return ret

  def compute_output(self, o, output_shape=None):
    """Compute output of multihead attention.

    Args:
      o: a Tensor with dimensions
         query_heads_dims + {value_dim} + other_dims
      output_shape: an optional Shape
    Returns:
      a Tensor with shape:
         {output_dim} + other_dims
    """
    if self.combine_dims:
      o = mtf.transpose(o, o.shape - self.o_dims + self.o_dims)
      o = mtf.replace_dimensions(o, self.o_dims, self.wo.shape.dims[-2])
      reduced_dims = [self.wo.shape.dims[-2]]
    else:
      reduced_dims = self.o_dims

    if self.keep_query_heads_dims:
      reduced_dims = [self.value_dim]

    return mtf.layers.us_einsum(
        [o, self.wo], output_shape=output_shape, reduced_dims=reduced_dims)

  @property
  def q_dims(self):
    return self.query_heads_dims + [self.key_dim]

  @property
  def k_dims(self):
    return self.memory_heads_dims + [self.key_dim]

  @property
  def v_dims(self):
    return self.memory_heads_dims + [self.value_dim]

  @property
  def o_dims(self):
    return self.query_heads_dims + [self.value_dim]


class ExpertsAttentionParams(AttentionParams):
  """Create attention parameters using experts-layer."""

  def __init__(self,
               mesh,
               query_input_dim,
               memory_input_dim,
               output_dim,
               key_dim,
               value_dim,
               query_heads_dims,
               memory_heads_dims,
               variable_dtype,
               shared_kv=False,
               no_query=False,
               combine_dims=True,
               ensemble_dim=None,
               keep_query_heads_dims=False,
               fold_scaling_into_initializer=True,
               context=None,
               experts_hparams=None,
               expert_computation="qkv"):
    super(ExpertsAttentionParams, self).__init__(
        mesh=mesh,
        query_input_dim=query_input_dim,
        memory_input_dim=memory_input_dim,
        output_dim=output_dim,
        key_dim=key_dim,
        value_dim=value_dim,
        query_heads_dims=query_heads_dims,
        memory_heads_dims=memory_heads_dims,
        variable_dtype=variable_dtype,
        shared_kv=shared_kv,
        no_query=no_query,
        combine_dims=combine_dims,
        ensemble_dim=ensemble_dim,
        keep_query_heads_dims=keep_query_heads_dims,
        fold_scaling_into_initializer=fold_scaling_into_initializer,
        make_attention_vars=False)

    self.context = context
    self.expert_computation = expert_computation

    # Unless we want to compute both q and kv, we can use the normal MoE
    # settings.
    if expert_computation == "qkv":
      experts_attention_compute_qkv = True
    elif expert_computation in ["q", "kv"]:
      experts_attention_compute_qkv = False
      if expert_computation == "q":
        # Always assume shared_kv.
        self.wkv = mtf.get_variable(
            self.mesh,
            "kv",
            self.k_shape,
            initializer=tf.random_normal_initializer(
                stddev=self.memory_input_dim.size ** -0.5),
            dtype=self.variable_dtype)
      else:  # Computing kv with experts.
        self.wq = mtf.get_variable(
            self.mesh,
            "q",
            self.q_shape,
            initializer=tf.random_normal_initializer(
                stddev=self.query_input_dim.size ** -0.5),
            dtype=self.variable_dtype)
    else:
      raise ValueError("Invalid expert computation mode: {}".format(
          expert_computation))

    # ExpertsAttention, for simplicitly, asserts that combine_dims is True, and
    # for efficiency, that shared_kv is True.
    if not self.combine_dims:
      raise ValueError("combine_dims must be True for ExpertsAttention.")
    if not self.shared_kv:
      raise ValueError("shared_kv must be True for ExpertsAttention.")
    if mtf.layers.unit_scaling_convention():
      raise NotImplementedError

    # Now replace "heads" dim with the "d_model" name to avoid conflicts when
    # we want to partition both "experts_hidden" and "heads".
    moe_output_dims = mtf.Dimension("d_model", self.q_shape[-1].size)

    tf.logging.info("ExpertsAttention moe_hidden_size: {}".format(
        experts_hparams.hidden_size))
    tf.logging.info("moe_output_dims: {}".format(moe_output_dims))
    self.moe_layer = mtf.transformer.moe.MoE1D(
        moe_gating=experts_hparams.moe_gating,
        num_experts=experts_hparams.num_experts,
        loss_coef=experts_hparams.loss_coef,
        group_size=experts_hparams.group_size,
        min_expert_capacity=experts_hparams.min_expert_capacity,
        capacity_factor_train=experts_hparams.capacity_factor_train,
        capacity_factor_eval=experts_hparams.capacity_factor_eval,
        switch_policy_train=experts_hparams.switch_policy_train,
        switch_policy_eval=experts_hparams.switch_policy_eval,
        switch_dropout=experts_hparams.switch_dropout,
        switch_temperature=experts_hparams.switch_temperature,
        switch_jitter=experts_hparams.switch_jitter,
        ntlb_top_k=experts_hparams.ntlb_top_k,
        hidden_size=experts_hparams.hidden_size,
        output_dim=moe_output_dims,
        use_experts_attention=experts_attention_compute_qkv,
        activation=experts_hparams.activation,
        z_loss=experts_hparams.z_loss)

  def _compute_merge_qkv(self, antecedent):
    """Computes qkv all in one call using MoE layer."""
    def _replace_d_model_dim(t):
      """Used to replace the `d_model` dim with `heads`."""
      new_last_dim = mtf.Dimension(self.q_shape[-1].name, t.shape[-1].size)
      return mtf.reshape(
          t, new_shape=mtf.Shape(t.shape[:-1] + [new_last_dim]))
    if self.expert_computation == "qkv":
      # NOTE: This assumes querty and memory antecedent are the same
      qk = self.moe_layer.call(self.context, antecedent)
      # Split qk here since they went through experts-layers
      q, k = qk
      q = _replace_d_model_dim(q)
      k = _replace_d_model_dim(k)
    elif self.expert_computation == "q":
      q = self.moe_layer.call(self.context, antecedent)
      q = _replace_d_model_dim(q)
      # Compute key/value normally
      k = mtf.layers.us_einsum(
          [antecedent, self.wkv], reduced_dims=[self.memory_input_dim])
    elif self.expert_computation == "kv":
      k = self.moe_layer.call(self.context, antecedent)
      k = _replace_d_model_dim(k)
      # Compute query normally
      q = mtf.layers.us_einsum(
          [antecedent, self.wq], reduced_dims=[self.query_input_dim])
    else:
      raise ValueError("Invalid expert computation mode: {}".format(
          self.expert_computation))

    # Scale query
    q *= self.key_dim.size ** -0.5
    self._q = mtf.replace_dimensions(q, q.shape.dims[-1], self.q_dims)
    self._k = mtf.replace_dimensions(k, k.shape.dims[-1], self.k_dims)

  def compute_q(self, query_antecedent):
    self._compute_merge_qkv(query_antecedent)
    return self._q

  def compute_k(self, memory_antecedent):
    del memory_antecedent
    return self._k

  def compute_kv(self, memory_antecedent):
    del memory_antecedent
    return self._k

  def compute_v(self, memory_antecedent):
    del memory_antecedent
    raise NotImplementedError("ExpertsAttention uses shared_kv = True.")


def _combined_dim(dims):
  return mtf.Dimension(dims[0].name, mtf.Shape(dims).size)


def attention_params_simple(
    mesh, io_dim, kv_dim, heads_dim, variable_dtype):
  """Common case attention parameters.

  Args:
    mesh: a Mesh
    io_dim: a Dimension (channels dimension of inputs and outputs)
    kv_dim: a Dimension (channels in keys and values)
    heads_dim: a Dimension (number of attention "heads")
    variable_dtype: a mtf.VariableDType
  Returns:
    an AttentionParams
  """
  return AttentionParams(
      mesh,
      query_input_dim=io_dim,
      memory_input_dim=io_dim,
      output_dim=io_dim,
      key_dim=kv_dim,
      value_dim=kv_dim,
      query_heads_dims=[heads_dim],
      memory_heads_dims=[heads_dim],
      variable_dtype=variable_dtype)


def local_attention_1d(q,
                       k,
                       v,
                       length_dim,
                       key_dim,
                       value_dim,
                       fully_autoregressive=True,
                       length_dim_num_splits=1,
                       radius=128,
                       sequence_id=1,
                       write_priority=None,
                       read_priority=None,
                       attention_kwargs=None):
  """Attention to the a neighborood around the source.

  If fully_autoregressive, then query position p can only see memory positions
  in the range (p - radius, p].

  If not fully_autoregressive, then query position p can only see memory
  positions in the range (p - window_size, p + radius].

  In addition, if write_priority and read_priority are provided, then attention
  is limited to position pairs where
  read_priority[query position] >= write_priority[memory position]

  Args:
    q: a Tensor containing length_dim
    k: a Tensor containing length_dim
    v: an optional Tensor containing length_dim.  If none then uses v=k.
    length_dim: a Dimension
    key_dim: a Dimension (the channels dimension of q and k)
    value_dim: a Dimension (the channels dimension of v)
    fully_autoregressive: a boolean
    length_dim_num_splits: an optional integer indicating how many ways the
      length dimension is split
    radius: an integer
    sequence_id: a Tensor or an integer
    write_priority: an optional Tensor containing length_dim
    read_priority: an optional Tensor containing length_dim
    attention_kwargs: optional keyword arguments for attention()

  Returns:
    a Tensor with the shape x.shape - key_dim + value_dim

  Raises:
    ValueError: if channels or depth don't match.
  """
  # Choose a suitable block size.
  # We choose the greatest divisor of length_per_split less than or equal
  # to max(window_size, 128)
  length_per_split = length_dim.size // length_dim_num_splits
  block_length = max(radius, 128)
  while length_per_split % block_length != 0:
    block_length -= 1
  query_block_length = mtf.Dimension("query_block_length", block_length)
  memory_block_length = mtf.Dimension("memory_block_length", block_length)
  # The num_blocks dimension gets the same name as the length dimension,
  # so it will be split in the same way.
  num_blocks = mtf.Dimension(length_dim.name, length_dim.size // block_length)
  def _reshape_query(x):
    return mtf.replace_dimensions(
        x, length_dim, [num_blocks, query_block_length])
  def _reshape_memory(x):
    x = mtf.replace_dimensions(
        x, length_dim, [num_blocks, memory_block_length])
    return (mtf.left_halo_exchange if fully_autoregressive
            else mtf.halo_exchange)(
                x, num_blocks, memory_block_length, radius)
  q = _reshape_query(q)
  k = _reshape_memory(k)
  if v:
    v = _reshape_memory(v)
  else:
    v = k
  if sequence_id is None:
    sequence_id = 1
  if (not isinstance(sequence_id, mtf.Tensor) or
      length_dim not in sequence_id.shape.dims):
    sequence_id += mtf.zeros(q.mesh, [length_dim], tf.int32)
  q_sequence_id = _reshape_query(sequence_id)
  m_sequence_id = _reshape_memory(sequence_id)
  pos = mtf.range(q.mesh, length_dim, dtype=tf.int32)
  q_pos = _reshape_query(pos)
  m_pos = _reshape_memory(pos)

  padded_memory_block_length = mtf.Dimension(
      "memory_block_length",
      (1 if fully_autoregressive else 2) * radius + block_length)

  relative_position = m_pos - q_pos
  visible = mtf.equal(q_sequence_id, m_sequence_id)
  visible = mtf.logical_and(visible, mtf.greater(relative_position, -radius))
  visible = mtf.logical_and(visible, mtf.less_equal(
      relative_position, 0 if fully_autoregressive else radius))
  if read_priority is not None:
    write_priority = _reshape_memory(write_priority)
    read_priority = _reshape_query(read_priority)
    visible = mtf.logical_and(
        visible, mtf.greater_equal(read_priority, write_priority))

  bias = visibility_mask_to_attention_bias(visible, q.dtype)
  o = attention(q, k, v, padded_memory_block_length,
                key_dim, value_dim, bias, **attention_kwargs)
  return mtf.replace_dimensions(o, [num_blocks, query_block_length], length_dim)


def visibility_mask_to_attention_bias(visible, dtype):
  """Convert a boolean visibility mask to an attention bias.

  The returned Tensor has large negative values in positions where
  visible=False.

  Args:
    visible: a boolean Tensor
    dtype: a dtype
  Returns:
    a Tensor with the given dtype and the same shape as "visible"
  """
  return mtf.cast(mtf.logical_not(visible), dtype) * -1e9


def maybe_reshape_attention_input_for_2d_sharding(
    context, q, k, v, bias, unsplittable_dims):
  """Reshape the inputs to attention to split over an unused mesh dimension.

  In the case where the attention computation is unnecessarily replicated,
  this function reshapes the attention inputs to remove the unnecessary
  replication.

  This becomes relevent when doing 2-dimenional model parallelism.
  d_model is sharded over one mesh dimension and [vocab, num_heads, d_ff] are
  sharded over the other mesh dimension.  This fully distributes all of the
  einsum operations, except for the internals of the attention computation.

  To distribute that computation, this function creates a new tensor-dimension
  from the low bits of either the batch dimension or the num_heads dimension,
  and then splits that dimension over the unused mesh dimension.

  Args:
    context: a transformer.Context
    q: a Tensor
    k: a Tensor
    v: a Tensor
    bias: a Tensor
    unsplittable_dims: a list of tensor-dimensions not to split.  The key/value
      dimensions should be passed here.
  Returns:
    reshaped_q: a Tensor
    reshaped_k: a Tensor
    reshaped_v: a Tensor
    reshaped_bias: a Tensor
  """
  original_inputs = q, k, v, bias
  # we need to know the layout and mesh-shape to figure out what to do.
  if not context or not context.model.layout or not context.model.mesh_shape:
    return original_inputs
  mesh_shape = mtf.convert_to_shape(context.model.mesh_shape)
  layout_rules = mtf.convert_to_layout_rules(context.model.layout)
  # find a mesh dim that is unused (no tensor-dimension is split across it)
  mesh_axis_used = [False] * mesh_shape.ndims
  for x in original_inputs:
    for mesh_axis in layout_rules.tensor_layout(
        x.shape, mesh_shape).tensor_axis_to_mesh_axis:
      if mesh_axis is not None:
        mesh_axis_used[mesh_axis] = True
  if False not in mesh_axis_used:
    return original_inputs
  mesh_dim = mesh_shape.dims[mesh_axis_used.index(False)]
  # Choose an appropriate name for the new tensor-dimension so that the layout
  #   will know to split it across the unused mesh dimension.
  tensor_dim_name = None
  tensor_dim_name = layout_rules.mesh_dimension_name_to_tensor_dimension_names(
      mesh_dim.name)
  if tensor_dim_name:
    tensor_dim_name = tensor_dim_name[0]
  else:
    return original_inputs
  # Find a tensor-dimension that we can further split, by breaking off the
  # lower bits into our new tensor-dimension.
  # This resplittable tensor-dimension must be presnent in all of q, k, v
  #   and must be large enough to be further split.
  resplittable_dim = None
  for d in q.shape.dims:
    if d in k.shape.dims and d in v.shape.dims and d not in unsplittable_dims:
      num_splits = mtf.tensor_dim_to_mesh_dim_size(
          context.model.layout, context.model.mesh_shape, d)
      if d.size % (num_splits * mesh_dim.size) == 0:
        resplittable_dim = d
        break
  if not resplittable_dim:
    return original_inputs
  new_dim_high = mtf.Dimension(resplittable_dim.name, num_splits)
  new_dim_low = mtf.Dimension(tensor_dim_name,
                              resplittable_dim.size // num_splits)
  def _my_reshape(x):
    if x and resplittable_dim in x.shape.dims:
      return mtf.replace_dimensions(
          x, resplittable_dim, [new_dim_high, new_dim_low])
    else:
      return x
  return _my_reshape(q), _my_reshape(k), _my_reshape(v), _my_reshape(bias)


def make_params_mtlprompt(task_num,
                          num_heads,
                          prefix_hidden_dim,
                          kv_dim,
                          context,
                          mtlprompt_share,
                          dropout_rate,
                          prompt_length=None):
  """Returns the parameters for MTL-Prompt mode."""

  tf.logging.info("MTL-Prompt mode is ON!")
  # Get task ids for the batch from the context cache.
  task_id = context.cache["task-id"]

  # Remove length dim from shape [batch_size, length] -> [batch_size].
  task_id = mtf.reshape(task_id, task_id.shape - task_id.shape.dims[-1])
  batch_size_dim = task_id.shape.get_dim_by_name("batch")

  task_num_dim = mtf.Dimension("task_num", task_num)
  prompt_length_dim = mtf.Dimension("memory_length", prompt_length)
  heads_dim = mtf.Dimension("heads", num_heads)
  prefix_hidden_dim = mtf.Dimension("prefix_hidden", prefix_hidden_dim)

  with tf.variable_scope("prompt"):
    # Initialize projection network matrices.

    # MTL-Prompt-Share mode.
    hidden_shape = [context.model.model_dim, prefix_hidden_dim]
    scratch_shape = [prefix_hidden_dim, heads_dim, kv_dim]
    if not mtlprompt_share:
      # MTL-Prompt-Sep mode.
      hidden_shape = [task_num_dim] + hidden_shape
      scratch_shape = [task_num_dim] + scratch_shape

    w_k_up = mtf.get_variable(
        context.mesh,
        "w_k_up",
        mtf.Shape(hidden_shape),
        dtype=context.variable_dtype)
    w_k_down = mtf.get_variable(
        context.mesh,
        "w_k_down",
        mtf.Shape(scratch_shape),
        dtype=context.variable_dtype)
    w_v_up = mtf.get_variable(
        context.mesh,
        "w_v_up",
        mtf.Shape(hidden_shape),
        dtype=context.variable_dtype)
    w_v_down = mtf.get_variable(
        context.mesh,
        "w_v_down",
        mtf.Shape(scratch_shape),
        dtype=context.variable_dtype)

  scope_name = tf.get_variable_scope().name
  scope = scope_name.split("/")[0]
  prompts = context.shared_params[scope]["prompts"]
  prompts = mtf.layers.layer_norm(
      prompts, dim=context.model.model_dim, name="prompts_layernorm")
  if context.train and dropout_rate != 0.0:
    prompts = mtf.dropout(prompts, context.train, 1.0 - dropout_rate)

  prompts_k_hidden = mtf.matmul(
      prompts,
      w_k_up,
      output_shape=[task_num_dim, prompt_length_dim, prefix_hidden_dim],
      reduced_dims=[context.model.model_dim])
  prompts_k_hidden = mtf.relu(prompts_k_hidden)
  if context.train and dropout_rate != 0.0:
    prompts_k_hidden = mtf.dropout(
        prompts_k_hidden, context.train, keep_prob=1.0 - dropout_rate)
  prompts_k = mtf.matmul(
      prompts_k_hidden,
      w_k_down,
      output_shape=[task_num_dim, prompt_length_dim, heads_dim, kv_dim],
      reduced_dims=[prefix_hidden_dim])

  prompts_v_hidden = mtf.matmul(
      prompts,
      w_v_up,
      output_shape=[task_num_dim, prompt_length_dim, prefix_hidden_dim],
      reduced_dims=[context.model.model_dim])
  prompts_v_hidden = mtf.relu(prompts_v_hidden)
  if context.train and dropout_rate != 0.0:
    prompts_v_hidden = mtf.dropout(
        prompts_v_hidden, context.train, keep_prob=1.0 - dropout_rate)
  prompts_v = mtf.matmul(
      prompts_v_hidden,
      w_v_down,
      output_shape=[task_num_dim, prompt_length_dim, heads_dim, kv_dim],
      reduced_dims=[prefix_hidden_dim])

  prompts_batch_k = mtf.gather(
      prompts_k,
      task_id,
      task_num_dim,
      output_shape=[batch_size_dim, prompt_length_dim, heads_dim, kv_dim])
  prompts_batch_v = mtf.gather(
      prompts_v,
      task_id,
      task_num_dim,
      output_shape=[batch_size_dim, prompt_length_dim, heads_dim, kv_dim])
  return prompts_batch_k, prompts_batch_v


def make_params_hyperprompt(task_num,
                            num_heads,
                            prefix_hidden_dim,
                            kv_dim,
                            context,
                            mtlprompt_share,
                            dropout_rate,
                            prompt_length,
                            scope=None):
  """Returns the parameters for HyperPrompt mode."""
  del mtlprompt_share
  tf.logging.info("HyperPrompt mode is ON!")
  # Get task ids for the batch from the context cache.
  task_id = context.cache["task-id"]

  # Remove length dim from shape [batch_size, length].
  task_id = mtf.reshape(task_id, task_id.shape - task_id.shape.dims[-1])

  batch_size_dim = [dim for dim in task_id.shape.dims if dim.name == "batch"][0]

  task_num_dim = mtf.Dimension("task_num", task_num)
  prompt_length_dim = mtf.Dimension("memory_length", prompt_length)
  heads_dim = mtf.Dimension("heads", num_heads)
  prefix_hidden_dim = mtf.Dimension("prefix_hidden", prefix_hidden_dim)

  prompts = context.shared_params[scope]["prompts"]
  prompts = mtf.layers.layer_norm(
      prompts, dim=context.model.model_dim, name="prompts_layernorm")
  if context.train and dropout_rate != 0.0:
    prompts = mtf.dropout(prompts, context.train, 1.0 - dropout_rate)

  scope_name = tf.get_variable_scope().name
  scope = scope_name.split("/")[0]

  # Get the layer id.
  layer_id = int(scope_name.split("/")[1].split("_")[1])

  task_raw_embeddings = context.shared_params[scope]["task_raw_embedding"]
  task_raw_embedding_dim = task_raw_embeddings.shape.dims[1]

  task_projector_layer_one = context.shared_params[scope][
      "task_projector_layer_one"]
  task_projector_layer_one_in_dim = task_projector_layer_one.shape.dims[0]
  task_hidden_dim = task_projector_layer_one.shape.dims[1]

  task_projector_layer_two = context.shared_params[scope][
      "task_projector_layer_two"]
  task_final_embedding_dim = task_projector_layer_two.shape.dims[1]

  layer_id_embeddings = context.shared_params[scope]["layer_embedding"]
  layer_num_dim = layer_id_embeddings.shape.dims[0]
  layer_id_embedding_dim = layer_id_embeddings.shape.dims[1]

  # Get the layer id embedding for the batch.
  if layer_id not in range(0, layer_num_dim.size):
    raise ValueError("encounter errors in parsing scope get layer_id.")
  layer_id_task_num = mtf.constant(
      task_raw_embeddings.mesh,
      layer_id,
      shape=mtf.Shape([task_num_dim]),
      dtype=tf.int32)
  layer_id_emb_task_num = mtf.gather(
      layer_id_embeddings,
      layer_id_task_num,
      layer_num_dim,
      output_shape=[task_num_dim, layer_id_embedding_dim])

  task_embeddings_concat = mtf.concat(
      [task_raw_embeddings, layer_id_emb_task_num],
      concat_dim_name=task_raw_embedding_dim.name)

  # Feed raw task-embedding to MLP to obtain the layer-aware task embedding.
  task_embeddings_concat_hidden = mtf.matmul(
      task_embeddings_concat,
      task_projector_layer_one,
      output_shape=[task_num_dim, task_hidden_dim],
      reduced_dims=[task_projector_layer_one_in_dim])
  task_embeddings_concat_hidden_relu = mtf.relu(task_embeddings_concat_hidden)

  if context.train and dropout_rate != 0.0:
    task_embeddings_concat_hidden_relu = mtf.dropout(
        task_embeddings_concat_hidden_relu,
        context.train,
        keep_prob=1.0 - dropout_rate)

  task_embeddings_layer_awared = mtf.matmul(
      task_embeddings_concat_hidden_relu,
      task_projector_layer_two,
      output_shape=[task_num_dim, task_final_embedding_dim],
      reduced_dims=[task_hidden_dim])

  task_embeddings_layer_awared = mtf.layers.layer_norm(
      task_embeddings_layer_awared,
      dim=task_final_embedding_dim,
      name="prompt_task_embed_layernorm")

  hypernet_w_k_up = context.shared_params[scope]["hypernet_w_k_up"]
  hypernet_w_k_down = context.shared_params[scope]["hypernet_w_k_down"]
  hypernet_w_v_up = context.shared_params[scope]["hypernet_w_v_up"]
  hypernet_w_v_down = context.shared_params[scope]["hypernet_w_v_down"]

  # Hypernetwork generates the prompts transformation
  w_k_up = mtf.matmul(
      task_embeddings_layer_awared,
      hypernet_w_k_up,
      output_shape=[task_num_dim, context.model.model_dim, prefix_hidden_dim],
      reduced_dims=[task_final_embedding_dim])

  w_k_down = mtf.matmul(
      task_embeddings_layer_awared,
      hypernet_w_k_down,
      output_shape=[task_num_dim, prefix_hidden_dim, heads_dim, kv_dim],
      reduced_dims=[task_final_embedding_dim])

  w_v_up = mtf.matmul(
      task_embeddings_layer_awared,
      hypernet_w_v_up,
      output_shape=[task_num_dim, context.model.model_dim, prefix_hidden_dim],
      reduced_dims=[task_final_embedding_dim])

  w_v_down = mtf.matmul(
      task_embeddings_layer_awared,
      hypernet_w_v_down,
      output_shape=[task_num_dim, prefix_hidden_dim, heads_dim, kv_dim],
      reduced_dims=[task_final_embedding_dim])

  prompts_k_hidden = mtf.matmul(
      prompts,
      w_k_up,
      output_shape=[task_num_dim, prompt_length_dim, prefix_hidden_dim],
      reduced_dims=[context.model.model_dim])
  prompts_k_hidden = mtf.relu(prompts_k_hidden)
  if context.train and dropout_rate != 0.0:
    prompts_k_hidden = mtf.dropout(
        prompts_k_hidden, context.train, keep_prob=1.0 - dropout_rate)
  prompts_k = mtf.matmul(
      prompts_k_hidden,
      w_k_down,
      output_shape=[task_num_dim, prompt_length_dim, heads_dim, kv_dim],
      reduced_dims=[prefix_hidden_dim])
  prompts_batch_k = mtf.gather(
      prompts_k,
      task_id,
      task_num_dim,
      output_shape=[batch_size_dim, prompt_length_dim, heads_dim, kv_dim])

  prompts_v_hidden = mtf.matmul(
      prompts,
      w_v_up,
      output_shape=[task_num_dim, prompt_length_dim, prefix_hidden_dim],
      reduced_dims=[context.model.model_dim])
  prompts_v_hidden = mtf.relu(prompts_v_hidden)
  if context.train and dropout_rate != 0.0:
    prompts_v_hidden = mtf.dropout(
        prompts_v_hidden, context.train, keep_prob=1.0 - dropout_rate)
  prompts_v = mtf.matmul(
      prompts_v_hidden,
      w_v_down,
      output_shape=[task_num_dim, prompt_length_dim, heads_dim, kv_dim],
      reduced_dims=[prefix_hidden_dim])
  prompts_batch_v = mtf.gather(
      prompts_v,
      task_id,
      task_num_dim,
      output_shape=[batch_size_dim, prompt_length_dim, heads_dim, kv_dim])

  return prompts_batch_k, prompts_batch_v


def concat_hyper_prompts_kv(k, v, scope_encoder_or_decoder, use_hyperprompt,
                            memory_length, task_num, num_heads,
                            prefix_hidden_dim, kv_dim, context, mtlprompt_share,
                            dropout_rate, prompt_length):
  """Performs the concatenation of hyper prompts to key and value."""
  # Inject hyper-prompts into keys and values.
  if use_hyperprompt:
    # HyperPrompt mode.
    prompts_batch_k, prompts_batch_v = make_params_hyperprompt(
        task_num,
        num_heads,
        prefix_hidden_dim,
        kv_dim,
        context,
        mtlprompt_share,
        dropout_rate,
        prompt_length=prompt_length,
        scope=scope_encoder_or_decoder)
  else:
    # MTL-Prompt mode.
    prompts_batch_k, prompts_batch_v = make_params_mtlprompt(
        task_num,
        num_heads,
        prefix_hidden_dim,
        kv_dim,
        context,
        mtlprompt_share,
        dropout_rate,
        prompt_length=prompt_length)

  k = mtf.concat([prompts_batch_k, k], concat_dim_name="memory_length")
  v = mtf.concat([prompts_batch_v, v], concat_dim_name="memory_length")
  memory_length = mtf.Dimension("memory_length",
                                memory_length.size + prompt_length)
  memory_position = mtf.range(context.mesh, memory_length, tf.int32)

  return k, v, memory_position, memory_length
