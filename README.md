# Radon Transformation

## Install
Clone repository and then:
	pip install .

## Usage
```
radon_op, fbp_op = get_operators(n_angles=200, det_count=500, image_size=400, device='cuda')
sino = radon_op(input) 						 # input size: bsz x 1 x image_size x image_size
reconstructed = fbp_op(sino)
```
