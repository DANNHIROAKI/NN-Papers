from functools import partial
import torch
from random import randint
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from sinkhorn_transformer.sinkhorn_transformer import SinkhornTransformerLM
from sinkhorn_transformer.autopadder import Autopadder

def default(value, default):
    return value if value is not None else default

def top_p(logits, thres = 0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cum_probs > (1 - thres)
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    sorted_logits[sorted_indices_to_remove] = float('-inf')
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)

def top_k(logits, thres = 0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

def pad_sequence_left(seqs, value):
    m = max([len(s) for s in seqs])
    return torch.stack([F.pad(s, (m - len(s), 0)) for s in seqs])

def random_truncate_inputs(inputs, mask = None, pad_value=0):
    b, t, device, dtype = *inputs.shape, inputs.device, inputs.dtype
    mask = default(mask, torch.ones_like(inputs))
    rand_lengths = torch.randint(2, t, (b, 1))
    rand_mask = (torch.arange(t) < rand_lengths).to(device)
    target_seqs = [t.masked_select(mask) for mask, t in zip(rand_mask, inputs)]
    mask_seqs = [m.masked_select(mask) for mask, m in zip(rand_mask, rand_mask)]
    return pad_sequence_left(target_seqs, pad_value), pad_sequence_left(mask_seqs, False)

class AutoregressiveWrapper(nn.Module):
    def __init__(self, net, ignore_index = None, pad_value = 0, pad_left = True):
        super().__init__()
        assert isinstance(net, SinkhornTransformerLM), 'generative trainer wrapper can only accept SinkhornTransformerLM class'
        self.pad_value = pad_value
        self.ignore_index = default(ignore_index, pad_value)

        self.net = Autopadder(net, pad_left = pad_left)
        self.max_seq_len = net.max_seq_len

    @torch.no_grad()
    def generate(self, start_tokens, seq_len, eos_token = None, temperature = 1., filter_logits_fn = top_k, filter_thres = 0.9, **kwargs):
        was_training = self.net.training
        num_dims = len(start_tokens.shape)

        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        b, t = start_tokens.shape

        self.net.eval()
        out = start_tokens
        input_mask = kwargs.pop('input_mask', None)

        if input_mask is None:
            input_mask = torch.full_like(out, True, dtype=torch.bool, device=out.device)

        for _ in range(seq_len):
            x = out[:, -self.max_seq_len:]
            input_mask = input_mask[:, -self.max_seq_len:]
            logits = self.net(x, input_mask=input_mask, **kwargs)[:, -1, :]
            filtered_logits = filter_logits_fn(logits, thres = filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim=-1)
            sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim=-1)
            input_mask = F.pad(input_mask, (0, 1), value=True)
            if eos_token is not None and (sample == eos_token).all():
                break

        out = out[:, t:]

        if num_dims == 1:
            out = out.squeeze(0)

        self.net.train(was_training)
        return out

    def forward(self, x, return_loss = False, randomly_truncate_sequence = False, **kwargs):
        pad = partial(pad_sequence, batch_first = True, padding_value = self.pad_value)

        if not return_loss:
            if not isinstance(x, torch.Tensor):
                x = pad(x)
            return self.net(x, **kwargs)

        m = kwargs.pop('input_mask', None)

        if randomly_truncate_sequence:
            x, m = random_truncate_inputs(x, m, pad_value = self.pad_value)

        if isinstance(x, torch.Tensor):
            xi, xo = x[:, :-1], x[:, 1:]
        else:
            xi = pad(list(map(lambda t: t[:-1], x)))
            xo = pad(list(map(lambda t: t[1:], x)))

        if m is not None:
            assert m.shape == x.shape[0:2], 'input mask must be the same shape as the input of the auto-regressive wrapper to automatically handle'
            kwargs.update(input_mask = m[:, :-1])

        out = self.net(xi, **kwargs)

        loss = F.cross_entropy(out.transpose(1, 2), xo, ignore_index = self.ignore_index)
        return loss
