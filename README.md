# Radon Transformation

## Install
- clone repository
- install pytorch: https://pytorch.org/get-started/locally/ 

	pip install ./radon_transformation/
## Usage
```
radon_op, fbp_op = get_operators(n_angles=200, det_count=500, image_size=400, device='cuda')
sino = radon_op(input) 						 # input size: bsz x 1 x image_size x image_size
reconstructed = fbp_op(sino)
```
