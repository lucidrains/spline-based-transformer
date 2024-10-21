import torch
from torch import nn, tensor
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

# uniform cubic b-spline

class BSpline(Module):
    def __init__(self):
        super().__init__()

        matrix = tensor([
            [-1,  3, -3,  1],
            [ 3, -6,  3,  0],
            [-3,  0,  3,  0],
            [ 1,  4,  1,  0]
        ]) / 6

        self.register_buffer('matrix', matrix)

    def forward(
        self,
        control_points,
        times
    ):
        batch, device = control_points.shape[0], control_points.device
        assert control_points.shape[1] == 4

        # just following the many b-spline equations i see online
        # open an issue if you see some obvious error

        powers = torch.arange(4, device = device)

        times = rearrange(times, 't -> t 1') ** powers
        times = repeat(times, '... -> b ...', b = batch)

        return times @ self.matrix @ control_points

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

        self.bspliner = BSpline()

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

    def decode_from_latents(
        self,
        control_points,
        num_times: int
    ):
        device = control_points.device

        # uniform times from 0 - 1

        times = torch.linspace(0, 1, num_times, device = device)

        splined_from_latent_controls = self.bspliner(control_points, times)

        recon = self.decoder(splined_from_latent_controls)
        return recon

    def forward(
        self,
        data,
        return_loss = False,
        return_latents = False
    ):
        batch, num_points, device = *data.shape[:2], data.device

        latents = repeat(self.control_point_latents, 'l d -> b l d', b = batch)

        encoder_input, unpack_fn = pack_with_inverse([latents, data], 'b * d')

        encoded = self.encoder(encoder_input)

        control_latents, encoded = unpack_fn(encoded)

        recon = self.decode_from_latents(control_latents, num_times = num_points)

        if not return_loss:
            if not return_latents:
                return recon

            return recon, control_latents

        recon_loss = F.mse_loss(recon, data)
        return recon_loss
