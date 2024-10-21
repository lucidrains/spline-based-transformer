import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import rearrange

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

# class

class SplineBasedTransformer(Module):
    def __init__(
        self,
        dim,
        enc_depth,
        dec_depth = None,
        dim_head = 64,
        heads = 8,
        dropout = 0.
    ):
        super().__init__()
        dec_depth = default(dec_depth, enc_depth)

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
        encoded = self.encoder(data)

        recon = self.decoder(encoded)

        if not return_loss:
            return recon

        recon_loss = F.mse_loss(recon, data)
        return recon_loss
