import math
import torch
import torch.nn as nn
import torch.nn.functional as tnf

from collections import OrderedDict
from compressai.layers import GDN
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models import get_scale_table
from compressai.models.utils import conv, deconv

from cloc import register_model
from cloc.models.base import BaseCompressor


def sinusoidal_embedding(values: torch.Tensor, dim=256, max_period=64):
    assert values.dim() == 1 and (dim % 2) == 0
    exponents = torch.linspace(0, 1, steps=(dim // 2))
    freqs = torch.pow(max_period, -1.0 * exponents).to(device=values.device)
    args = values.view(-1, 1) * freqs.view(1, dim//2)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding


class AdaptiveAffineBCHW(nn.Module):
    default_embedding_dim = 256 
    def __init__(self, dim, emb_dim=None):
        super().__init__()
        emb_dim = emb_dim or self.default_embedding_dim
        self.embedding_layer = nn.Sequential(
            nn.GELU(),
            nn.Linear(emb_dim, 2*dim),
            nn.Unflatten(1, unflattened_size=(2*dim, 1, 1))
        )
        self.embedding_layer[1].weight.data.mul_(0.01)
        self.embedding_layer[1].bias.data.zero_()

        self.variable_rate = True

    def forward(self, x, emb):
        shift, scale = self.embedding_layer(emb).chunk(chunks=2, dim=1)
        x = x * (1 + scale) + shift
        return x


class SequentialWithEmbedding(nn.Sequential):
    def forward(self, input, emb):
        for module in self:
            if getattr(module, 'variable_rate', False):
                input = module(input, emb)
            else:
                input = module(input)
        return input


def mlp(dims):
    return nn.Sequential(
        nn.Linear(dims[0], dims[1]),
        nn.GELU(),
        nn.Linear(dims[1], dims[1]),
    )


class MyHyperpriorModel(BaseCompressor):
    def __init__(self, N=192, M=320, lmb_range=(32, 1024), pretrain_lmb_range=None):
        super().__init__()

        # variable-rate modules
        _low, _high = lmb_range
        self.lmb_range = (float(_low), float(_high))
        self.default_lmb = self.lmb_range[1]
        self.lmb_embed_dim = (256, 256)
        self.lmb_embedding_enc = mlp(self.lmb_embed_dim)
        self.lmb_embedding_dec = mlp(self.lmb_embed_dim)
        self.lmb_embedding_em = mlp(self.lmb_embed_dim)
        self._sin_period = 64
        low, high = math.log(self.lmb_range[0]), math.log(self.lmb_range[1])
        self.default_eval_lmbs = torch.linspace(low, high, steps=4).exp().tolist()

        # for rate-incremental learning
        self.pretrain_lmb_range = lmb_range if (pretrain_lmb_range is None) else pretrain_lmb_range
        self.enc_lmb_clip = False
        self.dec_lmb_clip = False
        self.em_lmb_clip = False

        self.g_a = SequentialWithEmbedding(
            conv(3, N),
            GDN(N), AdaptiveAffineBCHW(N),
            conv(N, N),
            GDN(N), AdaptiveAffineBCHW(N),
            conv(N, N),
            GDN(N), AdaptiveAffineBCHW(N),
            conv(N, M),
        )
        self.g_s = SequentialWithEmbedding(
            deconv(M, N),
            GDN(N, inverse=True), AdaptiveAffineBCHW(N),
            deconv(N, N),
            GDN(N, inverse=True), AdaptiveAffineBCHW(N),
            deconv(N, N),
            GDN(N, inverse=True), AdaptiveAffineBCHW(N),
            deconv(N, 3),
        )
        self.h_a = SequentialWithEmbedding(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True), AdaptiveAffineBCHW(N),
            conv(N, N),
            nn.LeakyReLU(inplace=True), AdaptiveAffineBCHW(N),
            conv(N, N),
        )
        self.h_s = SequentialWithEmbedding(
            deconv(N, M),
            nn.LeakyReLU(inplace=True), AdaptiveAffineBCHW(M),
            deconv(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True), AdaptiveAffineBCHW(M * 3 // 2),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )
        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

        self.register_buffer('_dummy', torch.zeros(1), persistent=False)

    def freeze_weights(self, group='em'):
        if group == 'em':
            for p in self.lmb_embedding_em.parameters():
                p.requires_grad_(False)
            for p in self.entropy_bottleneck.parameters():
                p.requires_grad_(False)
            for p in self.h_s.parameters():
                p.requires_grad_(False)
            for p in self.gaussian_conditional.parameters():
                p.requires_grad_(False)
            # self.scale_lmb_embedding()
            self.em_lmb_clip = True
        elif group == 'dec':
            for p in self.parameters():
                p.requires_grad_(False)
            for p in self.g_a.parameters():
                p.requires_grad_(True)
            for p in self.h_a.parameters():
                p.requires_grad_(True)
            for p in self.lmb_embedding_enc.parameters():
                p.requires_grad_(True)
            self.em_lmb_clip = True
            self.dec_lmb_clip = True
        else:
            raise ValueError(f'Unknown group={group}')

    def scale_lmb_embedding(self):
        if self.lmb_range[1] > self.pretrain_lmb_range[1]: # rescale the lambda embedding layer
            factor = self.pretrain_lmb_range[1] / self.lmb_range[1]
            assert self.em_lmb_clip
            if not self.enc_lmb_clip:
                self.lmb_embedding_enc[0].weight.data.mul_(factor)
            if not self.dec_lmb_clip:
                self.lmb_embedding_dec[0].weight.data.mul_(factor)

    def sample_lmb(self, n: int):
        low, high = self.lmb_range # original lmb space, 16 to 1024
        p = 3.0
        low, high = math.pow(low, 1/p), math.pow(high, 1/p) # transformed space
        transformed_lmb = low + (high-low) * torch.rand(n, device=self._dummy.device)
        lmb = torch.pow(transformed_lmb, exponent=p)
        assert isinstance(lmb, torch.Tensor) and lmb.shape == (n,)
        return lmb

    def sin_embedding(self, lmb):
        scaled = torch.log(lmb) * self._sin_period / math.log(8192)
        return sinusoidal_embedding(scaled, self.lmb_embed_dim[0], max_period=self._sin_period)

    def get_lmb_embeddings(self, lmb: torch.Tensor):
        assert isinstance(lmb, torch.Tensor) and lmb.dim() == 1
        low, high = self.pretrain_lmb_range
        # entropy model
        lmb_em = torch.clamp(lmb, min=low, max=high) if self.em_lmb_clip else lmb
        lmb_emb_em = self.lmb_embedding_em(self.sin_embedding(lmb_em))
        # encoder
        lmb_enc = torch.clamp(lmb, min=low, max=high) if self.enc_lmb_clip else lmb
        lmb_emb_enc = self.lmb_embedding_enc(self.sin_embedding(lmb_enc))
        # decoder
        lmb_dec = torch.clamp(lmb, min=low, max=high) if self.dec_lmb_clip else lmb
        lmb_emb_dec = self.lmb_embedding_dec(self.sin_embedding(lmb_dec))
        return lmb_emb_em, lmb_emb_enc, lmb_emb_dec

    # def get_em_lmb_embedding(self, lmb: torch.Tensor):
    #     assert isinstance(lmb, torch.Tensor) and lmb.dim() == 1
    #     low, high = self.em_lmb_range
    #     lmb = torch.clamp(lmb, min=low, max=high)
    #     scaled = torch.log(lmb) * self._sin_period / math.log(8192)
    #     embedding = sinusoidal_embedding(scaled, self.lmb_embed_dim[0], max_period=self._sin_period)
    #     return self.lmb_embedding_em(embedding)

    # def get_encdec_lmb_embedding(self, lmb: torch.Tensor):
    #     assert isinstance(lmb, torch.Tensor) and lmb.dim() == 1
    #     scaled = torch.log(lmb) * self._sin_period / math.log(8192)
    #     embedding = sinusoidal_embedding(scaled, self.lmb_embed_dim[0], max_period=self._sin_period)
    #     return self.lmb_embedding_enc(embedding), self.lmb_embedding_dec(embedding)

    def forward(self, im):
        B, imC, imH, imW = im.shape # batch, channel, height, width
        x = self.preprocess(im)

        if self.training:
            lmb = self.sample_lmb(n=B)
        else:
            lmb = torch.full((B,), self.default_lmb, device=im.device)
        assert lmb.shape == (B,)
        lmb_emb_em, lmb_emb_enc, lmb_emb_dec = self.get_lmb_embeddings(lmb)

        y = self.g_a(x, lmb_emb_enc)
        z = self.h_a(y, lmb_emb_enc)

        z_hat, z_likelihoods = self.entropy_bottleneck(z)

        gaussian_params = self.h_s(z_hat, lmb_emb_em)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)

        x_hat = self.g_s(y_hat, lmb_emb_dec)

        # compute metrics
        bpp1 = -1.0 * torch.log2(z_likelihoods).sum(dim=(1,2,3)) / float(imH * imW)
        bpp2 = -1.0 * torch.log2(y_likelihoods).sum(dim=(1,2,3)) / float(imH * imW)
        bpp = bpp1 + bpp2
        mse = tnf.mse_loss(x_hat, x, reduction='none').mean(dim=[1, 2, 3])

        metrics = OrderedDict()
        assert bpp.shape == mse.shape == lmb.shape == (B,)
        loss = bpp + lmb * mse
        metrics['loss'] = loss.mean()

        with torch.inference_mode():
            metrics['bpp'] = bpp.mean().item()
            metrics['mse'] = mse.mean().item()
            im_hat = self.postprocess(x_hat)
            mse = tnf.mse_loss(im_hat, im, reduction='mean')
            metrics['psnr'] = -10.0 * torch.log10(mse).item()
            metrics['bpp1'] = bpp1.mean().item()
            metrics['bpp2'] = bpp2.mean().item()
        return metrics

    def get_latents(self, im, lmb):
        assert lmb.shape == im.shape[0:1]
        lmb_emb_em, lmb_emb_enc, lmb_emb_dec = self.get_lmb_embeddings(lmb)
        x = self.preprocess(im)

        y = self.g_a(x, lmb_emb_enc)
        z = self.h_a(y, lmb_emb_enc)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat, lmb_emb_em)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        return y_hat

    def decode_latents(self, latents, lmb):
        assert lmb.shape == latents.shape[0:1]
        lmb_emb_em, lmb_emb_enc, lmb_emb_dec = self.get_lmb_embeddings(lmb)
        x_hat = self.g_s(latents, lmb_emb_dec)
        im_hat = self.postprocess(x_hat)
        return im_hat

    def latents_replay(self, y_hat, lmb, im):
        lmb_emb_em, lmb_emb_enc, lmb_emb_dec = self.get_lmb_embeddings(lmb)
        x_hat = self.g_s(y_hat, lmb_emb_dec)

        x = self.preprocess(im)
        mse = tnf.mse_loss(x_hat, x, reduction='none').mean(dim=(1,2,3))
        loss = (lmb * mse).mean(0)
        return loss

    def prepare_compression(self):
        self.entropy_bottleneck.update(force=True)
        scale_table = get_scale_table()
        self.gaussian_conditional.update_scale_table(scale_table, force=True)

    def compress(self, im):
        lmb = torch.full(im.shape[0:1], self.default_lmb, device=im.device)
        lmb_emb_em, lmb_emb_enc, lmb_emb_dec = self.get_lmb_embeddings(lmb)

        x = self.preprocess(im)
        y = self.g_a(x, lmb_emb_enc)
        z = self.h_a(y, lmb_emb_enc)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat, lmb_emb_em)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        compressed_obj = ((y_strings, z_strings), (z.shape[-2], z.shape[-1]))
        return compressed_obj

    def decompress(self, compressed_obj):
        strings, shape = compressed_obj
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)

        lmb = torch.full(z_hat.shape[0:1], self.default_lmb, device=z_hat.device)
        lmb_emb_em, lmb_emb_enc, lmb_emb_dec = self.get_lmb_embeddings(lmb)

        gaussian_params = self.h_s(z_hat, lmb_emb_em)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        x_hat = self.g_s(y_hat, lmb_emb_dec)
        im_hat = self.postprocess(x_hat)
        return im_hat


@register_model
def msh_vr(lmb_range=(32, 1024), pretrain_lmb_range=None, pretrained=False):
    model = MyHyperpriorModel(lmb_range=lmb_range, pretrain_lmb_range=pretrain_lmb_range)

    if pretrained is True:
        url = 'https://huggingface.co/duanzh0/my-model-weights/resolve/main/cloc/msh_vr.pt'
        checkpoint = torch.hub.load_state_dict_from_url(url, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    elif pretrained: # str or Path
        checkpoint = torch.load(pretrained)
        model.load_state_dict(checkpoint['model'])
    return model


@register_model
def msh_vr_frzdec(lmb_range=(32, 1024), pretrain_lmb_range=(32, 1024), pretrained=True):
    model = MyHyperpriorModel(lmb_range=lmb_range, pretrain_lmb_range=pretrain_lmb_range)

    if pretrained is True:
        url = 'https://huggingface.co/duanzh0/my-model-weights/resolve/main/cloc/msh_vr.pt'
        checkpoint = torch.hub.load_state_dict_from_url(url, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    elif pretrained: # str or Path
        checkpoint = torch.load(pretrained)
        model.load_state_dict(checkpoint['model'])
    model.freeze_weights(group='dec')
    return model


@register_model
def msh_vr_frzem(lmb_range=(32, 1024), pretrain_lmb_range=(32, 1024), pretrained=True):
    model = MyHyperpriorModel(lmb_range=lmb_range, pretrain_lmb_range=pretrain_lmb_range)

    if pretrained is True:
        url = 'https://huggingface.co/duanzh0/my-model-weights/resolve/main/cloc/msh_vr.pt'
        checkpoint = torch.hub.load_state_dict_from_url(url, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    elif pretrained: # str or Path
        checkpoint = torch.load(pretrained)
        model.load_state_dict(checkpoint['model'])
    model.freeze_weights(group='em')
    return model
