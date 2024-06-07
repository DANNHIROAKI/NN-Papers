import torch.nn as nn
from torch.nn.functional import *
import torch
from collections import namedtuple
import torch_blocksparse
import torch_blocksparse_cpp_utils
import sys

def multi_head_attention_forward(query,                           # type: Tensor
                                 key,                             # type: Tensor
                                 value,                           # type: Tensor
                                 embed_dim_to_check,              # type: int
                                 num_heads,                       # type: int
                                 in_proj_weight,                  # type: Tensor
                                 in_proj_bias,                    # type: Tensor
                                 bias_k,                          # type: Optional[Tensor]
                                 bias_v,                          # type: Optional[Tensor]
                                 add_zero_attn,                   # type: bool
                                 dropout_p,                       # type: float
                                 out_proj_weight,                 # type: Tensor
                                 out_proj_bias,                   # type: Tensor
                                 sparse_dot_sdd_nt,               # type: Callable
                                 sparse_dot_dsd_nn,               # type: Callable
                                 sparse_softmax,                  # type: Callable
                                 training=True,                   # type: bool
                                 key_padding_mask=None,           # type: Optional[Tensor]
                                 need_weights=True,               # type: bool
                                 attn_mask=None,                  # type: Optional[Tensor]
                                 use_separate_proj_weight=False,  # type: bool
                                 q_proj_weight=None,              # type: Optional[Tensor]
                                 k_proj_weight=None,              # type: Optional[Tensor]
                                 v_proj_weight=None,              # type: Optional[Tensor]
                                 static_k=None,                   # type: Optional[Tensor]
                                 static_v=None,                   # type: Optional[Tensor]
                                 key_padding_mask_mode='add',     # type: String ['add', 'mul']
                                 attn_mask_mode='add'             # type: String ['add', 'mul']
                                 ):
    # type: (...) -> Tuple[Tensor, Optional[Tensor]]

    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    assert key.size() == value.size()

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if not use_separate_proj_weight:
        if torch.equal(query, key) and torch.equal(key, value):
            # self-attention
            q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

        elif torch.equal(key, value):
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim:(embed_dim * 2)])
            v = linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2):])
        else:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask,
                                      torch.zeros((attn_mask.size(0), 1),
                                                  dtype=attn_mask.dtype,
                                                  device=attn_mask.device)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros((key_padding_mask.size(0), 1),
                                                   dtype=key_padding_mask.dtype,
                                                   device=key_padding_mask.device)], dim=1)
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if attn_mask is not None:
        assert attn_mask.size(0) == src_len
        assert attn_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = torch.cat([attn_mask, torch.zeros((attn_mask.size(0), 1),
                                                          dtype=attn_mask.dtype,
                                                          device=attn_mask.device)], dim=1)
        if key_padding_mask is not None:
            key_padding_mask = torch.cat(
                [key_padding_mask, torch.zeros((key_padding_mask.size(0), 1),
                                               dtype=key_padding_mask.dtype,
                                               device=key_padding_mask.device)], dim=1)

    q = q.view(bsz, num_heads, q.shape[1], q.shape[2])
    k = k.view(bsz, num_heads, k.shape[1], k.shape[2])
    v = v.view(bsz, num_heads, v.shape[1], v.shape[2])
    # attention scores
    attn_output_weights = sparse_dot_sdd_nt(q, k)
    attn_output_weights = sparse_softmax(attn_output_weights, key_padding_mask=key_padding_mask, attn_mask=attn_mask,
                                         key_padding_mask_mode=key_padding_mask_mode, attn_mask_mode=attn_mask_mode)
    # outputs
    attn_output = sparse_dot_dsd_nn(attn_output_weights, v)
    attn_output = attn_output.view(bsz*num_heads, attn_output.shape[2], attn_output.shape[3])
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
    return attn_output, None


make_layout = torch_blocksparse_cpp_utils.make_layout

class MultiheadAttention(nn.modules.activation.MultiheadAttention):

    ops = dict()
    # add to cache
    def get_ops(self, L):
        import sys
        if L not in MultiheadAttention.ops:
            sparse_dot_sdd_nt = torch_blocksparse.MatMul(self.layout, self.block, 'sdd', trans_a=False, trans_b=True)
            sparse_dot_dsd_nn = torch_blocksparse.MatMul(self.layout, self.block, 'dsd', trans_a=False, trans_b=False)
            sparse_softmax = torch_blocksparse.Softmax(self.layout, self.block)
            MultiheadAttention.ops[L] = (sparse_dot_sdd_nt, sparse_dot_dsd_nn, sparse_softmax)
        return MultiheadAttention.ops[L]

    # constructor
    def __init__(self, embed_dim, num_heads, layout, block, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None,
                 key_padding_mask_mode='add', attn_mask_mode='mul'):
        if dropout != 0:
            raise NotImplementedError('dropout is not supported for now')

        super(MultiheadAttention, self).__init__(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim)
        self.layout = layout
        self.block = block
        self.key_padding_mask_mode = key_padding_mask_mode
        self.attn_mask_mode = attn_mask_mode


    # forward pass
    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None):
        # check that operation is supported
        if query.shape != key.shape or key.shape != value.shape:
            raise NotImplementedError('only self-attention is supported for now')
        # cache look-up table computations etc
        sparse_dot_sdd_nt, sparse_dot_dsd_nn, sparse_softmax = self.get_ops(query.shape[0])
        # execute
        if not self._qkv_same_embed_dim:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                sparse_dot_sdd_nt, sparse_dot_dsd_nn, sparse_softmax,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                key_padding_mask_mode=self.key_padding_mask_mode, attn_mask_mode=self.attn_mask_mode)
        else:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                sparse_dot_sdd_nt, sparse_dot_dsd_nn, sparse_softmax,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask,
                key_padding_mask_mode=self.key_padding_mask_mode, attn_mask_mode=self.attn_mask_mode)

def replace_mha(model, info):
    for child_name, child in model.named_children():
        if isinstance(child, nn.modules.activation.MultiheadAttention):
            add_bias_kv = child.bias_k is not None
            device = child.in_proj_weight.device
            mha = MultiheadAttention(child.embed_dim, child.num_heads, info,
                                     dropout=child.dropout, add_bias_kv=add_bias_kv, 
                                     add_zero_attn=child.add_zero_attn, kdim=child.kdim,
                                     vdim=child.vdim, key_padding_mask_mode='mul', 
                                     attn_mask_mode='add').to(device)
            for yparam, xparam in zip(mha.parameters(), child.parameters()):
                yparam.data.copy_(xparam.data)
            setattr(model, child_name, mha)
        else:
            replace_mha(child, info)
