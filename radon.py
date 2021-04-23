'''
    File name: radon.py
    Author: Hannah Dröge
    Date created: 4/22/2021
    Python Version: 3.6
'''

import torch
import numpy as np

# for dataset
from dival import get_standard_dataset

# for displaying
import matplotlib.pyplot as plt

class radon(torch.nn.Module):
    '''
    Radon Transformation
    '''
    def __init__(self, n_angles, det_count, image_size, step_size=1, sampling_points=100, batchify_rays=[], device='cuda'):
        super(radon, self).__init__()
        self.sample_points = self.get_sample_points(n_angles, det_count, image_size, step_size, sampling_points)
        self.batchify_rays = batchify_rays
        self.device = device

    def forward(self, input):
        n_steps, n_dims, n_angles, n_count = self.sample_points.shape
        sample_points = self.sample_points.reshape([n_steps, n_dims, -1])
        if len(self.batchify_rays)>0:
            sample_points = sample_points[:,:,self.batchify_rays]
        sample_points = sample_points.unsqueeze(-1)
        out = self.intensity_from_rays(sample_points, input, self.device)
        out = torch.rot90(out, 2)

        out = out.view(n_angles, n_count)
        return out

    def intensity_from_rays(self,sample_points, img, device):
        '''
        Calculates Intenity I_d of Radon transformation: I_d = exp(sum of pixel at alpha shift: u(x)){*I_0}
        Args:
            param sample_points [# steps, axis, angles, shifts]: Sample Points in Image.
            param img [w,h]: 2d input image.
        
        Returns the intensities in form of [angles * shifts]
        '''
        img_n = img[None, None]
        grid = sample_points.permute(0,2,3,1)
        steps, angles, shifts, axis = grid.shape
        grid = grid.reshape(1,-1,1,2).to(device)
        out = torch.nn.functional.grid_sample(img_n, grid, padding_mode='zeros', align_corners=True)
        out = out.view(steps, angles, shifts)
        intens = torch.sum(out, dim = 0) / out.shape[0]
        return intens

    def get_sample_points(self, n_angles, det_count, image_size, step_size=1, sampling_points=100):
        ''' 
        Calculates the samples, sampled by radon transformation.
        Args:
            param n_angles: Number of measuring angles between[0,pi].
            param det_count: Number of rays that will be projected.
            image_size: Size of the measured image.
            step_size: Step size for sampling.
            sampling_points: number of sampling points.

        Returns the coordinates in form of [# steps, axis, angles, shifts]
        '''
        distance = 2 / image_size
        sx = torch.linspace(-1, 1, det_count).unsqueeze(0)
        sx = sx * det_count / 2 * distance
        sx = sx * step_size

        sy = -np.sqrt(2)

        ex = sx
        ey = -sy

        angles = torch.linspace(0, np.pi, n_angles).unsqueeze(-1) + (np.pi / 2)

        rsx = sx * torch.cos(angles) - sy * torch.sin(angles) # start x
        rsy = sx * torch.sin(angles) + sy * torch.cos(angles) # start y
        rex = ex * torch.cos(angles) - ey * torch.sin(angles) # end x
        rey = ex * torch.sin(angles) + ey * torch.cos(angles) # end y
        rdx = rex - rsx # ray x
        rdy = rey - rsy # ray y

        start = torch.stack((rsx,rsy))
        ray = torch.stack((rdx,rdy))

        n_steps = sampling_points

        coords = torch.zeros((n_steps, 2, n_angles, det_count))

        rand_shifts = (torch.rand(n_steps) * 1/n_steps) - ((1/2) * 1/n_steps)


        for i in range(n_steps):
            coords[i] = start + i * (1 / n_steps) * ray

        return coords


if __name__=='__main__':

    device = 'cuda'

    # Load dataset
    dataset = get_standard_dataset('lodopab')
    data= dataset.create_torch_dataset('train')
    idx = 0
    fd_reconstruction = torch.tensor(data[idx][1]).to(device)
    ld_sinogram =  torch.tensor(data[idx][0]).to(device)

    # Apply radon
    radon_op = radon(1000, 500, fd_reconstruction.shape[0],device=device)
    out = radon_op(fd_reconstruction)

    print("Input: ", fd_reconstruction.shape)
    print("Radon Transform: ", out.shape)

    plt.imshow(out.cpu())
    plt.savefig('output.png')
    plt.close()
    
