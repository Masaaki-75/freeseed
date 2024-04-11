import torch
import numpy as np
import torch.nn as nn
from torch_radon import Radon, RadonFanbeam
import torch.nn.functional as F

"""
The wrappers are used to provide methods for preparing sparse-view input data.
"""


class BasicSparseWrapper(nn.Module):
    def __init__(self, num_views=None, img_size=256, num_full_views=720, source_distance=1075, det_count=672):
        super().__init__()
        self.num_full_views = num_full_views
        self.source_distance = source_distance
        self.det_count = det_count
        self.img_size = img_size
        if num_views is None:
            raise ValueError('num_views not provided, need an integer value, e.g. 18/36/72/144.')
        else:
            self.num_views = num_views
            print(f'full views: {self.num_full_views}, sparse views: {num_views}.')
    
    # ------------ basic radon function ----------------
    # avoid possible cuda error, put radon func in the module
    def radon(self, sinogram, num_views=None, angle_bias=0):
        '''sinogram to ct image'''
        angles = self.get_angles(num_views, angle_bias)
        radon_tool = RadonFanbeam(self.img_size, angles, self.source_distance, det_count=self.det_count,)
        filter_sin = radon_tool.filter_sinogram(sinogram, "ram-lak")
        back_proj = radon_tool.backprojection(filter_sin) 
        return back_proj 
    
    def image_radon(self, ct_image, num_views=None, angle_bias=0):
        '''ct image to sinogram'''
        angles = self.get_angles(num_views, angle_bias)
        radon_tool = RadonFanbeam(self.img_size, angles, self.source_distance, det_count=self.det_count,)
        sinogram = radon_tool.forward(ct_image)
        return sinogram
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError('forward in wrapper should be implemented.')

    # ------------ basic sparse-view CT data generation ----------------
    def generate_sparse_and_full_ct(self, mu_ct, angle_bias=0):
        sparse_sinogram, full_sinogram = self.generate_sparse_and_full_sinogram(mu_ct, angle_bias)
        full_mu = self.radon(full_sinogram,)
        sparse_mu = self.radon(sparse_sinogram, self.num_views, angle_bias)
        return sparse_mu, full_mu
    
    def generate_sparse_and_full_sinogram(self, mu_ct, angle_bias=0):
        full_sinogram = self.image_radon(mu_ct)
        sparse_sinogram = self.image_radon(mu_ct, self.num_views, angle_bias)
        return sparse_sinogram, full_sinogram
    
    def get_angles(self, num_views=None, angle_bias=0, is_bias_radian=True):
        num_views = self.num_full_views if num_views is None else num_views  # specified number of views
        angles = np.linspace(0, np.pi*2, num_views, endpoint=False)  # select views according to the specified number of views
        angle_bias = angle_bias / 360 * 2 * np.pi if not is_bias_radian else angle_bias
        # print('angles:', angles / (np.pi * 2) * 360)
        # print('angle_bias:', angle_bias / (np.pi * 2)* 360)
        return angles + angle_bias
        # For example: 
            # If
            #   angles = [0, 10, 20, ..., 170, 180] / 180 * pi
            #   bias = 5 / 180 * pi
            # Then angles + bias is then a shifted/biased version of the selected views.

    
    # ------------ dual-domain sparse-view CT data generation ----------------
    def generate_sparse_and_full_dudo(self, mu_ct, return_sinomask=False, mixed_interp=False):
        full_sinogram = self.image_radon(mu_ct)
        sparse_sinogram = self.image_radon(mu_ct, self.num_views)
        full_mu = self.radon(full_sinogram)
        # sinogram_full = self.add_noise(sinogram_full)
        
        sparse_mask_vec = self.generate_sparse_sinogram_mask(self.num_views)  # [Nv]
        
        bs = full_sinogram.shape[0]
        num_det = full_sinogram.shape[-1]
        sparse_mask = sparse_mask_vec.reshape(1, 1, len(sparse_mask_vec), 1)  # [1, 1, Nv]
        sparse_mask = sparse_mask.repeat(bs, 1, 1, num_det).float()  # [B, 1, Nv, Nd]
        
        if mixed_interp:
            interp_sinogram = F.interpolate(sparse_sinogram, size=full_sinogram.shape[2:], mode='bilinear')
            sparse_sinogram = interp_sinogram #* (1 - sparse_mask) + sparse_mask * full_sinogram
        else:
            sparse_sinogram = sparse_mask * full_sinogram
            
        sparse_sinogram_reduce = full_sinogram.permute(0,1,3,2)[..., sparse_mask_vec != 0].permute(0,1,3,2).contiguous()
        sparse_mu = self.radon(sparse_sinogram_reduce, num_views=self.num_views)
        
        if return_sinomask:
            return sparse_sinogram, sparse_mu, full_sinogram, full_mu, sparse_mask
        else:
            return sparse_sinogram, sparse_mu, full_sinogram, full_mu
    
    def generate_sparse_sinogram_mask(self, num_views):
        sinogram_mask = np.arange(1, self.num_full_views+1)
        sinogram_mask = np.ma.masked_equal(sinogram_mask % (self.num_full_views // num_views), 1)
        sinogram_mask = sinogram_mask.mask.astype(np.int32)  # [Nv,]
        sinogram_mask = torch.from_numpy(sinogram_mask).cuda()
        return sinogram_mask

    # ------------ dual-domain sparse-view CT data generation ----------------
    def generate_multiview_mu_list(self, mu_ct, num_views_list):
        multiview_mu_list = [None] * len(num_views_list)
        for i, num_views in enumerate(num_views_list):
            num_views = int(num_views)
            sinogram = self.image_radon(mu_ct, num_views)
            multiview_mu_list[i] = self.radon(sinogram, num_views)
        return multiview_mu_list
    
    def generate_multiview_sino_list(self, mu_ct, num_views_list):
        multiview_sino_list = [None] * len(num_views_list)
        for i, num_views in enumerate(num_views_list):
            num_views = int(num_views)
            sinogram = self.image_radon(mu_ct, num_views)
            multiview_sino_list[i] = sinogram
        return multiview_sino_list

    def generate_sparse_and_full_list(self, mu_ct,):
        """
        Generate:
            - a list of sparse-view CT images of same sparse-view patterns (but with different number of views!).
            - full-view CT image.
        """
        multiview_mu_list = self.generate_multiview_mu_list(mu_ct)
        sparse_mu_list = multiview_mu_list[:-1]
        full_mu = multiview_mu_list[-1]
        
        return sparse_mu_list, full_mu

