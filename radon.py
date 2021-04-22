'''
    File name: radon.py
    Author: Hannah Dröge
    Date created: 4/22/2021
    Python Version: 3.6
'''

import torch
import numpy as np

# for dataset
import dival
from dival import get_standard_dataset

# for displaying
import matplotlib.pyplot as plt

class radon(torch.nn.Module):
    '''
    Radon Transformation
    '''
    def forward(self, input, settings):
        with torch.no_grad():
            n_angles   = settings["n_angles"]
            det_count  = settings["det_count"]
            image_size = settings["image_size"]
            step_size = settings["step_size"]
            sampling_points = settings["sampling_points"]
            batchify_rays = settings["batchify_rays"]
   
        sample_points = self.get_sample_points(n_angles, det_count, image_size, step_size, sampling_points)

        n_steps, n_dims, n_angles, n_count = sample_points.shape
        sample_points = sample_points.reshape([n_steps, n_dims, -1])
        if len(batchify_rays)>0:
            sample_points = sample_points[:,:,batchify_rays]
        sample_points = sample_points.unsqueeze(-1)
        out = self.intensity_from_rays(sample_points, input)
        out = torch.rot90(out, 2)

        out = out.view(n_angles, n_count)
        return out

    def intensity_from_rays(self,sample_points, img):
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
        grid = grid.reshape(1,-1,1,2)
        out = torch.nn.functional.grid_sample(img_n, grid.cuda(), padding_mode='zeros', align_corners=True)
        out = out.view(steps, angles, shifts)
        intens = torch.sum(out, dim = 0) / out.shape[0]
        

        return intens

    def get_sample_points(self, n_angles, det_count, image_size, step_size = 1, sampling_points = 100):
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


    # Load dataset
    dataset = get_standard_dataset('lodopab')
    data= dataset.create_torch_dataset('train')
    idx = 0
    fd_reconstruction = torch.tensor(data[idx][1],device='cuda')
    ld_sinogram = torch.tensor(data[idx][0],device='cuda')

    # Transformation settings
    settings = {
        "n_angles"   : 1000,
        "det_count"  : 500,
        "image_size" : fd_reconstruction.shape[0],
        "step_size" : 1,
        "sampling_points" : 100,
        "batchify_rays": [],
    }


    # Apply radon
    radon_op = radon()
    out = radon_op(fd_reconstruction,settings)

    print("Input: ", fd_reconstruction.shape)
    print("Radon Transform: ", out.shape)

    plt.imshow(out.cpu())
    plt.savefig('Output.png')
    plt.imshow(ld_sinogram.cpu())
    plt.savefig('ld_sinogram.png')
