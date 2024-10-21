import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import rearrange, repeat, pack, unpack

from x_transformers import (
    Encoder,
    RMSNorm,
    FeedForward
)

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def pack_with_inverse(t, pattern):
    t, packed_shape = pack(t, pattern)

    def inverse(t, inverse_pattern = None):
        inverse_pattern = default(inverse_pattern, pattern)
        return unpack(t, packed_shape, inverse_pattern)

    return t, inverse

# class

class SplineBasedTransformer(Module):
    def __init__(
        self,
        dim,
        enc_depth,
        dec_depth = None,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        num_control_points = 4
    ):
        super().__init__()
        dec_depth = default(dec_depth, enc_depth)

        self.control_point_latents = nn.Parameter(torch.zeros(num_control_points, dim))

        self.encoder = Encoder(
            dim = dim,
            heads = heads,
            depth = enc_depth,
            attn_dim_head = dim_head,
            attn_dropout = dropout,
            ff_dropout = dropout
        )

        self.decoder = Encoder(
            dim = dim,
            heads = heads,
            depth = dec_depth,
            attn_dim_head = dim_head,
            attn_dropout = dropout,
            ff_dropout = dropout
        )

    def forward(
        self,
        data,
        return_loss = False
    ):
        batch = data.shape[0]

        latents = repeat(self.control_point_latents, 'l d -> b l d', b = batch)

        encoder_input, unpack_fn = pack_with_inverse([latents, data], 'b * d')

        encoded = self.encoder(encoder_input)

        latents, encoded = unpack_fn(encoded)

        recon = self.decoder(encoded)

        if not return_loss:
            return recon

        recon_loss = F.mse_loss(recon, data)
        return recon_loss
