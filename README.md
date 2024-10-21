<img src="./spline-based-transformer.png" width="400px"></img>

## Spline-Based Transformer (wip)

Implementation of the proposed <a href="https://www.youtube.com/watch?v=AzolLlIbKhg">Spline-Based Transformer</a> from Disney Research

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

recon = model(data)
assert data.shape == recon.shape
```

## Citations

```bibtex
@misc{Chandran2024,
    author  = {Prashanth Chandran, Agon Serifi, Markus Gross, Moritz Bächer},
    url     = {https://la.disneyresearch.com/publication/spline-based-transformers/}
}
```
