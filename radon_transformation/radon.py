'''
    File name: radon.py
    Author: Hannah Dröge
    Date created: 4/22/2021
    Python Version: 3.6
'''

from numpy.core.fromnumeric import repeat
import torch
import numpy as np

# for dataset
from skimage.data import shepp_logan_phantom
from skimage.transform.radon_transform import iradon
from skimage.transform.radon_transform import radon as radon_nm

from radon_transformation.filter import rampfilter
# for displaying
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class radon(torch.nn.Module):
    '''
    Radon Transformation
    '''
    def __init__(self, n_angles, det_count, image_size, sampling_points=None, batchify_rays=[], device='cuda'):
        super(radon, self).__init__()
        self.sample_points = self.get_sample_points(n_angles, det_count, image_size, sampling_points).to(device)
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

        out = out.view(out.shape[0],1, n_angles, n_count)
        return out#.permute(0,1,3,2)

    def intensity_from_rays(self,sample_points, img, device):
        '''
        Calculates Intenity I_d of Radon transformation: I_d = exp(sum of pixel at alpha shift: u(x)){*I_0}
        Args:
            sample_points [# steps, axis, angles, shifts]: Sample Points in Image.
            img [w,h]: 2d input image.
        
        Returns the intensities in form of [angles * shifts]
        '''
        img_n = img#[None, None]
        bsz = img.shape[0]
        grid = sample_points.permute(0,2,3,1)
        steps, angles, shifts, axis = grid.shape
        grid = grid.reshape(1,-1,1,2).repeat(bsz,1,1,1)
        out = torch.nn.functional.grid_sample(img_n, grid, padding_mode='zeros', align_corners=True)
        out = out.view(bsz, steps, angles, shifts)
        intens = torch.sum(out, dim = 1) #/ out.shape[0]
        return intens

    @staticmethod
    def get_sample_points(n_angles, det_count, image_size, sampling_points=None):
        ''' 
        Calculates the samples, sampled by radon transformation.
        Args:
            n_angles: Number of measuring angles between[0,pi].
            det_count: Number of rays that will be projected.
            image_size: Size of the measured image.
            sampling_points: number of sampling points.

        Returns the coordinates in form of [# steps, axis, angles, shifts]
        '''
        step_size = image_size/det_count
        distance = 2 / image_size
        sx = torch.linspace(-1, 1, det_count).unsqueeze(0)
        sx = sx * det_count / 2 * distance
        sx = sx * step_size

        sy = -np.sqrt(2)

        ex = sx
        ey = -sy

        angles = torch.linspace(0, np.pi, n_angles).unsqueeze(-1) #+ (np.pi / 2)

        rsx = sx * torch.cos(angles) - sy * torch.sin(angles) # start x
        rsy = sx * torch.sin(angles) + sy * torch.cos(angles) # start y
        rex = ex * torch.cos(angles) - ey * torch.sin(angles) # end x
        rey = ex * torch.sin(angles) + ey * torch.cos(angles) # end y
        rdx = rex - rsx # ray x
        rdy = rey - rsy # ray y

        start = torch.stack((rsx,rsy))
        ray = torch.stack((rdx,rdy))

        if sampling_points == None:
            n_steps = int(image_size * np.sqrt(2))
        else:
            n_steps = sampling_points

        coords = torch.zeros((n_steps, 2, n_angles, det_count))

        rand_shifts = (torch.rand(n_steps) * 1/n_steps) - ((1/2) * 1/n_steps)

        for i in range(n_steps):
            coords[i] = start + i * (1 / n_steps) * ray

        return coords

    
class fbp(torch.nn.Module):
    ''' 
    Filtered Backprojection
    '''
    def __init__(self, n_angles=1000, det_count=400, image_size=400, device="cuda"):
        super().__init__()
        self.image_size=image_size
        self.step_size = image_size/det_count
        self.det_count = det_count
        self.n_angles = n_angles
        

        # padding values
        projection_size_padded = max(64, int(2 ** (2 * torch.tensor(det_count)).float().log2().ceil()))
        self.pad_width = (projection_size_padded - det_count)

        #filter
        self.filter = rampfilter(projection_size_padded).to(device)

        # create grid

        # get angles 
        thetas = torch.linspace(0, np.pi-(np.pi/n_angles), n_angles)[:,None,None]
        

        # get grid [-1,1]
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1,1,image_size), torch.linspace(-1,1,image_size))

        # get rotated grid
        tgrid = (grid_x*thetas.cos() - grid_y*thetas.sin()).unsqueeze(-1)

        y = torch.ones_like(tgrid) * torch.linspace(-1,1,n_angles)[:,None,None,None]

        self.grid = torch.cat((y,tgrid),dim=-1).view(self.n_angles * self.image_size, self.image_size, 2)[None].to(device)
        



    def forward(self, input):

        input=input.permute(0,1,3,2)
        bsz, _, det_count, _ = input.shape
        input = input.double()

        # pad input
        padded_input = torch.nn.functional.pad(input, [0, 0, 0, self.pad_width], mode='constant', value=0)

        # apply filter
        projection = torch.fft.fft(padded_input,dim=2) * self.filter[:,None].double()
        radon_filtered = torch.real(torch.fft.ifft(projection,dim=2))[:, :, :det_count, :]

        # reconstruct
        grid = self.grid.repeat(bsz,1,1,1).double()
        reconstructed = torch.nn.functional.grid_sample(radon_filtered, grid, mode="bilinear", padding_mode='zeros', align_corners=True)
        reconstructed = reconstructed.view(bsz, self.n_angles, 1, self.image_size, self.image_size).sum(1)
        reconstructed = reconstructed/self.step_size
        return reconstructed  * np.pi / (2 * self.n_angles)

def get_operators(n_angles=380, det_count=500, image_size=400, device='cuda'):
    radon_op = radon(n_angles=n_angles, det_count=det_count, image_size=image_size, device=device)
    fbp_op = fbp(n_angles=n_angles, det_count=det_count, image_size=image_size, device=device)
    return radon_op, fbp_op


if __name__=='__main__':

    device = 'cuda'


    fd_reconstruction = torch.tensor(shepp_logan_phantom()).cuda().float() # Shape: 400 x 400

    fd_reconstruction=torch.cat((fd_reconstruction[None,None],fd_reconstruction[None,None]),dim=0)

    # Apply radon
    radon_op, fbp_op = get_operators(n_angles=380, det_count=500, image_size=400, device=device)


    sino = radon_op(fd_reconstruction)
    reconstructed = fbp_op(sino)


    plt.figure(figsize=(20,10))

    plt.subplot(131)
    plt.imshow(fd_reconstruction[0,0].cpu())
    plt.title("Input")
    plt.colorbar()


    plt.subplot(132)
    plt.imshow(reconstructed.cpu()[0,0])
    plt.title("Padded")
    plt.colorbar()


    plt.subplot(133)
    plt.imshow(sino[0,0].cpu())
    plt.title("sino")
    plt.colorbar()


    plt.savefig("comparison.png")


