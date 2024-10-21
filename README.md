<img src="./spline-based-transformer.png" width="400px"></img>

## Spline-Based Transformer (wip)

Implementation of the proposed <a href="https://www.youtube.com/watch?v=AzolLlIbKhg">Spline-Based Transformer</a> ([paper](https://la.disneyresearch.com/wp-content/uploads/SBT.pdf)) from Disney Research

This is basically a transformer based autoencoder, but they cleverly use a set of latent tokens, where that set of tokens are the (high dimensional) control points for a spline.

## Install

```bash
$ pip install spline-based-transformer
```

## Usage

```python
import torch
from spline_based_transformer import SplineBasedTransformer

model = SplineBasedTransformer(
    dim = 512,
    enc_depth = 6,
    dec_depth = 6
)

data = torch.randn(1, 1024, 512)

loss = model(data, return_loss = True)
loss.backward()

# after much training

recon, control_points = model(data, return_latents = True)
assert data.shape == recon.shape

# mess with the control points, which should preserve continuity better

control_points += 1

controlled_recon = model.decode_from_latents(control_points, num_times = 1024)
assert controlled_recon.shape == data.shape
```

## Citations

```bibtex
@misc{Chandran2024,
    author  = {Prashanth Chandran, Agon Serifi, Markus Gross, Moritz Bächer},
    url     = {https://la.disneyresearch.com/publication/spline-based-transformers/}
}
```
