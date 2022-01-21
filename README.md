# Radon Transformation



A Pytorch implementation of the radon operator and filtered backprojection with, except for a constant, adjoint radon operator and backprojection.

![Alt text](./comparison.png?raw=true "")

## Install
- clone repository
- install pytorch: https://pytorch.org/get-started/locally/

```
pip install ./Radon-Transformation/
```

## Usage
```
from radon_transformation.radon import get_operators

input = # Load some input of shape (bsz x 1 x w x h)

radon_op, fbp_op = get_operators(n_angles=200, det_count=500, image_size=input.shape[-1], device='cuda')
sino = radon_op(input)
reconstructed = fbp_op(sino)
```
