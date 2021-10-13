'''
    File name: radon.py
    Author: Hannah Dröge
    Date created: 4/22/2021
    Python Version: 3.6
'''
import torch
import numpy as np

# for dataset
from skimage.data import shepp_logan_phantom
from skimage.transform.radon_transform import iradon
from skimage.transform.radon_transform import radon as radon_nm

from radon_transformation.filter import rampfilter
import matplotlib.pyplot as plt


class radon(torch.nn.Module):
    ''' 
    Radon Transformation
    '''
    def __init__(self, n_angles=1000, image_size=400, device="cuda"):
        super(radon, self).__init__()
        self.n_angles=n_angles

        # get angles 
        thetas = torch.linspace(0, np.pi-(np.pi/n_angles), n_angles)[:,None,None].to(device)
        cos_al, sin_al = thetas.cos(), thetas.sin()
        zeros = torch.zeros_like(cos_al)

        # calculate rotations
        rotations = torch.stack((cos_al,sin_al,zeros,-sin_al, cos_al,zeros),-1).reshape(-1,2,3)
        self.rotated = torch.nn.functional.affine_grid(rotations, torch.Size([n_angles, 1, image_size, image_size]), align_corners=True).reshape(1,-1,image_size,2)


    def forward(self, image):
        bsz, _, shape_size, _ = image.shape
        out_fl = torch.nn.functional.grid_sample(image, self.rotated.repeat(bsz,1,1,1), align_corners=True).reshape(bsz,1,self.n_angles,shape_size,shape_size)
        out = out_fl.sum(3).permute(0,1,3,2)
        return out




class fbp(torch.nn.Module):
    ''' 
    Filtered Backprojection
    '''
    def __init__(self, n_angles=1000, image_size=400, circle = False, filtered=True, device="cuda"):
        super().__init__()
        self.image_size=image_size
        det_count = image_size
        self.step_size = image_size/det_count
        self.n_angles = n_angles
        self.circle=circle
        self.filtered=filtered

        # padding values
        projection_size_padded = max(64, int(2 ** (2 * torch.tensor(det_count)).float().log2().ceil()))
        self.pad_width = (projection_size_padded - det_count)

        #filter
        self.filter = rampfilter(projection_size_padded).to(device)

        # get angles 
        thetas = torch.linspace(0, np.pi-(np.pi/n_angles), n_angles)[:,None,None] 

        # get grid [-1,1]
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1,1,image_size), torch.linspace(-1,1,image_size))

        # get rotated grid
        tgrid = (grid_x*thetas.cos() - grid_y*thetas.sin()).unsqueeze(-1)
        y = torch.ones_like(tgrid) * torch.linspace(-1,1,n_angles)[:,None,None,None]
        self.grid = torch.cat((y,tgrid),dim=-1).view(self.n_angles * self.image_size, self.image_size, 2)[None].to(device)

        self.reconstruction_circle = (grid_x ** 2 + grid_y ** 2) <= 1

    def forward(self, input):

        bsz, _, det_count, _ = input.shape
        input = input.double()
        if self.filtered:
            # pad input
            padded_input = torch.nn.functional.pad(input, [0, 0, 0, self.pad_width], mode='constant', value=0)

            # apply filter
            projection = torch.fft.fft(padded_input,dim=2) * self.filter[:,None].double()
            radon_filtered = torch.real(torch.fft.ifft(projection,dim=2))[:, :, :det_count, :]
        else:
            radon_filtered = input
        # reconstruct
        grid = self.grid.repeat(bsz,1,1,1).double()
        reconstructed = torch.nn.functional.grid_sample(radon_filtered, grid, mode="bilinear", padding_mode='zeros', align_corners=True)
        reconstructed = reconstructed.view(bsz, self.n_angles, 1, self.image_size, self.image_size).sum(1)
        reconstructed = reconstructed/self.step_size

        # circle
        if self.circle:
            reconstructed_circle = self.reconstruction_circle.repeat(bsz,1,1,1).double()
            reconstructed[reconstructed_circle==0] = 0.

        return reconstructed  * np.pi / (2 * self.n_angles)

def get_operators(n_angles=380, image_size=400, device='cuda'):
    radon_op = radon(n_angles=n_angles, image_size=image_size, device=device)
    fbp_op = fbp(n_angles=n_angles, image_size=image_size, circle=False, device=device)
    return radon_op, fbp_op


def test_adjoint():
    n_angles = 50
    image_size = 100
    device = 'cpu'

    radon_op = radon(n_angles=n_angles, image_size=image_size, device=device)
    fbp_op = fbp(n_angles=n_angles, image_size=image_size, circle=False, device=device, filtered=False)

    x = torch.rand([1,1,100,100]).to(device)
    y = torch.rand([1,1,100,50]).to(device)

    leftside = torch.sum(radon_op(x) * y).item()
    rightside = torch.sum(x * fbp_op(y)).item()

    print("\n<Ax,y>=", leftside,"  -----  <x,A'y>=", rightside)
    print('\n leftside/rightside: ',leftside/rightside)
    return leftside/rightside



if __name__=='__main__':

    device = 'cuda'
    torch.manual_seed(7)
    mini = 100
    maxi = 0
    for i in range(1):
        val = test_adjoint()
        if val < mini:
            mini = val
        if val > maxi:
            maxi = val
    #print(mini-maxi)

    # Create Input
    fd_reconstruction = torch.tensor(shepp_logan_phantom()).cuda().float() # Shape: 400 x 400
    fd_reconstruction=torch.cat((fd_reconstruction[None,None],fd_reconstruction[None,None]),dim=0)

    # Apply 
    radon_op, fbp_op = get_operators(n_angles=750, image_size=400, device=device)
    sino = radon_op(fd_reconstruction.cuda())
    reconstructed = fbp_op(sino)

    # Plot
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


