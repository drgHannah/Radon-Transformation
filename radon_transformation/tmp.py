import numpy as np
import matplotlib.pyplot as plt

from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale
from skimage.transform import iradon

#np.random.seed(8)

x = np.random.rand(100,100).astype(np.double)
y = np.random.rand(142,450).astype(np.double)

theta = np.linspace(0., 180., 450, endpoint=False)
leftside = np.sum(radon(x, theta=theta, circle=False) * y)
rightside = np.sum(x * iradon(y, theta=theta, filter_name=None, circle=False, interpolation='linear'))#, preserve_range=True))#*31.83

print("\n<Ax,y>=", leftside,"  -----  <x,A'y>=", rightside)
print('\n leftside/rightside: ',leftside/rightside)



# class radon(torch.nn.Module):
#     '''
#     Radon Transformation
#     '''
#     def __init__(self, n_angles, det_count, image_size, sampling_points=None, batchify_rays=[], device='cuda'):
#         super(radon, self).__init__()
#         self.sample_points = self.get_sample_points(n_angles, det_count, image_size, sampling_points).to(device)
#         self.batchify_rays = batchify_rays
#         self.device = device

#     def forward(self, input):
#         n_steps, n_dims, n_angles, n_count = self.sample_points.shape
#         sample_points = self.sample_points.reshape([n_steps, n_dims, -1])
#         if len(self.batchify_rays)>0:
#             sample_points = sample_points[:,:,self.batchify_rays]
#         sample_points = sample_points.unsqueeze(-1)
#         out = self.intensity_from_rays(sample_points, input, self.device)
#         out = torch.rot90(out, 2)

#         out = out.view(out.shape[0],1, n_angles, n_count)
#         return out#.permute(0,1,3,2)

#     def intensity_from_rays(self,sample_points, img, device):
#         '''
#         Calculates Intenity I_d of Radon transformation: I_d = exp(sum of pixel at alpha shift: u(x)){*I_0}
#         Args:
#             sample_points [# steps, axis, angles, shifts]: Sample Points in Image.
#             img [w,h]: 2d input image.
        
#         Returns the intensities in form of [angles * shifts]
#         '''
#         img_n = img#[None, None]
#         bsz = img.shape[0]
#         grid = sample_points.permute(0,2,3,1)
#         steps, angles, shifts, axis = grid.shape
#         grid = grid.reshape(1,-1,1,2).repeat(bsz,1,1,1)
#         out = torch.nn.functional.grid_sample(img_n, grid, padding_mode='zeros', align_corners=True)
#         out = out.view(bsz, steps, angles, shifts)
#         intens = torch.sum(out, dim = 1) #/ out.shape[0]
#         return intens

#     @staticmethod
#     def get_sample_points(n_angles, det_count, image_size, sampling_points=None):
#         ''' 
#         Calculates the samples, sampled by radon transformation.
#         Args:
#             n_angles: Number of measuring angles between[0,pi].
#             det_count: Number of rays that will be projected.
#             image_size: Size of the measured image.
#             sampling_points: number of sampling points.

#         Returns the coordinates in form of [# steps, axis, angles, shifts]
#         '''
#         step_size = image_size/det_count
#         distance = 2 / image_size
#         sx = torch.linspace(-1, 1, det_count).unsqueeze(0)
#         sx = sx * det_count / 2 * distance
#         sx = sx * step_size

#         sy = -np.sqrt(2)

#         ex = sx
#         ey = -sy

#         angles = torch.linspace(0, np.pi, n_angles).unsqueeze(-1) #+ (np.pi / 2)

#         rsx = sx * torch.cos(angles) - sy * torch.sin(angles) # start x
#         rsy = sx * torch.sin(angles) + sy * torch.cos(angles) # start y
#         rex = ex * torch.cos(angles) - ey * torch.sin(angles) # end x
#         rey = ex * torch.sin(angles) + ey * torch.cos(angles) # end y
#         rdx = rex - rsx # ray x
#         rdy = rey - rsy # ray y

#         start = torch.stack((rsx,rsy))
#         ray = torch.stack((rdx,rdy))

#         if sampling_points == None:
#             n_steps = int(image_size * np.sqrt(2))
#         else:
#             n_steps = sampling_points

#         coords = torch.zeros((n_steps, 2, n_angles, det_count))

#         rand_shifts = (torch.rand(n_steps) * 1/n_steps) - ((1/2) * 1/n_steps)

#         for i in range(n_steps):
#             coords[i] = start + i * (1 / n_steps) * ray

#         return coords