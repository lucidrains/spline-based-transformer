import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from x_transformers import (
    RMSNorm,
    Encoder
)

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# class

class SplineBasedTransformer(Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        data,
        return_loss = False
    ):
        recon = data.clone()
        return data
