import torch
from torch import nn, Tensor, tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import rearrange, repeat, pack, unpack

from x_transformers import Encoder
from x_transformers.x_transformers import AlibiPositionalBias

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

# a simple mlp for encoding data and decoding

def MLP(dim, dim_out = None, expand_factor = 2):
    dim_out = default(dim_out, dim)
    dim_inner = dim_out * expand_factor

    return nn.Sequential(
        nn.Linear(dim, dim_inner),
        nn.GELU(),
        nn.Linear(dim_inner, dim_out)
    )

# uniform cubic b-spline

class BSpline(Module):
    def __init__(
        self,
        learned = False
    ):
        super().__init__()

        matrix = tensor([
            [-1,  3, -3,  1],
            [ 3, -6,  3,  0],
            [-3,  0,  3,  0],
            [ 1,  4,  1,  0]
        ]) / 6

        self.coeff = nn.Parameter(matrix, requires_grad = learned)

    def forward(
        self,
        control_points: Tensor,
        num_times: int
    ):
        batch, device = control_points.shape[0], control_points.device
        assert control_points.shape[1] == 4

        # uniform times from 0 - 1

        times = torch.linspace(0, 1, num_times, device = device)

        # following https://en.wikipedia.org/wiki/B-spline
        # open an issue if you see some obvious error

        powers = torch.arange(4, device = device).flip(dims = (0,))

        times = repeat(times, 't -> b t 1', b = batch) ** powers

        return times @ self.coeff @ control_points

# class

class SplineBasedTransformer(Module):
    def __init__(
        self,
        dim,
        enc_depth,
        model_dim = None,
        dec_depth = None,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        num_control_points = 4,
        encoder_kwargs: dict = dict(),
        decoder_kwargs: dict = dict(),
    ):
        super().__init__()
        model_dim = default(model_dim, dim)
        dec_depth = default(dec_depth, enc_depth)

        self.num_control_points = num_control_points
        self.control_point_latents = nn.Parameter(torch.zeros(num_control_points, dim))

        self.bspliner = BSpline()

        self.mlp_in = MLP(dim, model_dim)

        self.alibi = AlibiPositionalBias(heads) # todo - figure out if the paper accounted for asymmetric slopes given alibi weakness in non-causal setting

        self.encoder = Encoder(
            dim = dim,
            heads = heads,
            depth = enc_depth,
            attn_dim_head = dim_head,
            attn_dropout = dropout,
            ff_dropout = dropout,
            **encoder_kwargs
        )

        self.to_control_points = nn.Linear(dim, dim)

        self.decoder = Encoder(
            dim = dim,
            heads = heads,
            depth = dec_depth,
            attn_dim_head = dim_head,
            attn_dropout = dropout,
            ff_dropout = dropout,
            **decoder_kwargs
        )

        self.mlp_out = MLP(model_dim, dim)

    def decode_from_latents(
        self,
        control_points: Tensor,
        num_times: int,
        attn_bias = None
    ):
        device = control_points.device

        splined_from_latent_controls = self.bspliner(control_points, num_times)

        if not exists(attn_bias):
            attn_bias = self.alibi(num_times, num_times)

        decoded = self.decoder(splined_from_latent_controls, attn_bias = attn_bias)

        recon = self.mlp_out(decoded)
        return recon

    def forward(
        self,
        data: Tensor,
        return_loss = False,
        return_latents = False
    ):
        batch, num_points, device = *data.shape[:2], data.device

        data = self.mlp_in(data)

        # prepare control point latents across all batch samples

        latents = repeat(self.control_point_latents, 'l d -> b l d', b = batch)

        encoder_input, unpack_fn = pack_with_inverse([latents, data], 'b * d')

        # prepare alibi attention bias, but encoder is a bit different, as there should be no relative distance to the control latents

        attn_bias = self.alibi(num_points, num_points)
        encoder_attn_bias = F.pad(attn_bias, (self.num_control_points, 0, self.num_control_points, 0), value = 0.) # there is no relative distance between the latents and all the data points

        # encode

        encoded = self.encoder(encoder_input, attn_bias = encoder_attn_bias)

        # splice out control latents

        latents, encoded = unpack_fn(encoded)

        control_points = self.to_control_points(latents)

        # reconstruct data from the bottleneck

        recon = self.decode_from_latents(control_points, num_times = num_points, attn_bias = attn_bias)

        if not return_loss:
            if not return_latents:
                return recon

            return recon, control_points

        recon_loss = F.mse_loss(recon, data)
        return recon_loss
