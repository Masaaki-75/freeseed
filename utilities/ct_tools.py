import torch
import numpy as np
import sys
sys.path.append('..')
from utilities.tomo_tools_astra import Fanbeam2d
#import numba as nb

class CTTransforms(Fanbeam2d):
    def __init__(
        self, 
        img_shape, 
        full_angles,
        mu_water=0.192,  #  0.192 [1/cm] = 192 [1/m]
        mu_air=0.0,
        source_distance=1075, 
        num_dets=672, 
        gpu_idx=0, 
        recon_mask=None, 
        det_distance=None, 
        det_spacing=None, 
        simul_poisson_rate=-1,
        simul_gaussian_rate=-1,
        min_hu=-1024,
        max_hu=3072
        ):
        super().__init__(img_shape, source_distance, num_dets, gpu_idx, recon_mask, det_distance, det_spacing)
        self.mu_water = mu_water
        self.mu_air = mu_air
        self.full_angles = np.array(full_angles)
        self.simul_poisson_rate = simul_poisson_rate
        self.simul_gaussian_rate = simul_gaussian_rate
        self.min_hu = min_hu
        self.max_hu = max_hu
    
    @staticmethod
    def equidist_sampling(elems, num_samples, sampling_axis=0):
        num_elems = np.array(elems).shape[sampling_axis]
        indices = np.linspace(0, num_elems-1, num_samples, False)
        indices = np.floor(indices).astype(int)
        if sampling_axis == 0:
            return elems[indices]
        elif sampling_axis == 1:
            return elems[:, indices]
        elif sampling_axis == 2:
            return elems[:, :, indices]
        else:
            raise NotImplementedError("Currently not implemented with sampling_axis outside of (0,1,2).")
    
    def get_indices_from_full_angles(self, ranges):
        """Get the corresponding indices of `angles` as a subset of `full_angles`."""
        # `ranges` is a list/tuple containing ranges of the selected views
        # For example, ranges = [(0, np.pi/6), (np.pi/3, np.pi/2)]
        # selects items from `self.full_angles` satisfying 
        #   0 <= items <= pi/6  or  pi/3 <= items <= pi/2
        selected_indices = []
        full_angles = self.full_angles
        for (min_val, max_val) in ranges:
            s = np.where(np.logical_and(full_angles >= min_val, full_angles <= max_val))[0]
            selected_indices.extend(s)
        return sorted(selected_indices)
    
    def low_dose_transform(self, sinogram, poisson_rate, gaussian_rate):
        sinogram = self.add_poisson_noise_to_sinogram(sinogram, noise_rate=poisson_rate)
        sinogram = self.add_gaussian_noise_to_sinogram(sinogram, sigma=gaussian_rate)
        return sinogram
    
    def sparse_view_transform(self, sinogram, num_views):
        # sinogram here is of the shape [#views, #dets]
        return self.equidist_sampling(sinogram, num_views, sampling_axis=0)
    
    def limited_angle_transform(self, sinogram, ranges):
        indices = self.get_indices_from_full_angles(ranges)
        return sinogram[indices]
    
    ### Standard preprocessing functions ###
    def get_simulated_sinogram_from_hu(self, hu_img, use_numpy=True, on_cuda=False):
        hu_img = self.clip_range(hu_img, input_hu=True, min_val=self.min_hu, max_val=self.max_hu)
        mu_img = np.float32(self.hu2mu(hu_img))
        sinogram = self.fp(mu_img, self.full_angles, use_gpu=True)
        
        if use_numpy:
            sinogram = self.add_poisson_noise_to_sinogram_numpy(sinogram, self.simul_poisson_rate)
            sinogram = self.add_gaussian_noise_to_sinogram_numpy(sinogram, self.simul_gaussian_rate)
        else:
            sinogram = torch.from_numpy(sinogram).float().cuda() if on_cuda else sinogram
            sinogram = self.add_poisson_noise_to_sinogram(sinogram, self.simul_poisson_rate)
            sinogram = self.add_gaussian_noise_to_sinogram(sinogram, self.simul_gaussian_rate)
            sinogram = sinogram.detach().cpu().numpy()
        return sinogram
    
    def get_simulated_image_from_hu(self, hu_img, use_numpy=True, on_cuda=False):
        sinogram = self.get_simulated_sinogram_from_hu(hu_img, use_numpy=use_numpy, on_cuda=on_cuda)
        image = self.fbp(sinogram, self.full_angles, use_gpu=True)
        return image
    
    
    def hu2mu(self, hu_img):
        mu_img = hu_img / 1000 * (self.mu_water - self.mu_air) + self.mu_water
        return mu_img

    def mu2hu(self, mu_img):
        hu_img = (mu_img - self.mu_water) / (self.mu_water - self.mu_air) * 1000
        return hu_img

    def clip_range(self, img, input_hu=True, min_val=-1024, max_val=3072):
        assert min_val < max_val
        img = self.mu2hu(img) if not input_hu else img
        img = img.clip(min_val, max_val)
        img = self.hu2mu(img) if not input_hu else img
        return img
    
    def get_windowed_hu(self, hu_img, width=3000, center=500, norm_to_255=False):
        ''' HU_img -> 0-1 normalization'''
        window_min = float(center) - 0.5 * float(width)
        win_img = (hu_img - window_min) / float(width)
        win_img[win_img < 0] = 0
        win_img[win_img > 1] = 1
        if norm_to_255:
            win_img = (win_img * 255).astype('float')
        return win_img

    def get_unwindowed_hu(self, win_img, width=3000, center=500, norm=False):
        ''' 0-1 normalization -> HU_img'''
        window_min = float(center) - 0.5 * float(width)
        if norm:
            win_img = win_img / 255.
        hu_img = win_img * float(width) + window_min
        return hu_img
    
    ### Noise simulation functions ###
    @staticmethod
    def add_poisson_noise_to_sinogram_numpy(sinogram, noise_rate=1e6):
        # noise rate: background intensity or source influx
        if not isinstance(sinogram, np.ndarray):
            #print(f'Input should be ndarray, got {type(sinogram)} (auto-converted).')
            sinogram = np.array(sinogram, dtype=np.float32)
            
        if noise_rate > 0:
            max_val = sinogram.max() * 10 / 67  # this factor makes the noise more realistic?
            sinogram_ct = noise_rate * np.exp(-sinogram / max_val)
            # add poison noise
            sinogram_noise = np.random.poisson(sinogram_ct).clip(min=sinogram_ct.min())
            sinogram_out = - max_val * np.log(sinogram_noise / noise_rate)
        else:
            sinogram_out = sinogram
        return sinogram_out.astype(np.float32)
    
    @staticmethod
    def add_poisson_noise_to_sinogram(sinogram, noise_rate=1e6):
        # noise rate: background intensity or source influx
        if not torch.is_tensor(sinogram):
            #print(f'Input should be tensor, got {type(sinogram)} (auto-converted).')
            if isinstance(sinogram, np.ndarray):
                sinogram = torch.from_numpy(sinogram)
            else:
                sinogram = torch.tensor(sinogram)
            
        if noise_rate > 0:
            max_val = sinogram.max() * 10 / 67  # this factor makes the noise more realistic?
            sinogram_ct = noise_rate * torch.exp(-sinogram / max_val)
            # add poison noise
            sinogram_noise = torch.poisson(sinogram_ct).clamp(min=sinogram_ct.min())
            sinogram_out = - max_val * torch.log(sinogram_noise / noise_rate)
        else:
            sinogram_out = sinogram
        return sinogram_out
    
    @staticmethod
    def add_gaussian_noise_to_sinogram_numpy(sinogram, sigma=0.1):
        if not isinstance(sinogram, np.ndarray):
            #print(f'Input should be ndarray, got {type(sinogram)} (auto-converted).')
            sinogram = np.array(sinogram)
        
        if sigma > 0:
            shape = sinogram.shape
            sinogram_out = sinogram + sigma * np.random.randn(*shape)
            sinogram_out = sinogram_out.clip(sinogram.min(), sinogram.max())
        else:
            sinogram_out =  sinogram
        return sinogram_out
    
    @staticmethod
    def add_gaussian_noise_to_sinogram(sinogram, sigma=0.1):
        if not torch.is_tensor(sinogram):
            #print(f'Input should be tensor, got {type(sinogram)} (auto-converted).')
            if isinstance(sinogram, np.ndarray):
                sinogram = torch.from_numpy(sinogram)
            else:
                sinogram = torch.tensor(sinogram)
        
        if sigma > 0:
            dtype = sinogram.dtype
            if not sinogram.is_floating_point():
                sinogram = sinogram.to(torch.float32)    
            sinogram_out = sinogram + sigma * torch.randn_like(sinogram)
            sinogram_out = sinogram_out.clamp(sinogram.min(), sinogram.max())
            if sinogram_out.dtype != dtype:
                sinogram_out = sinogram_out.to(dtype)
        else:
            sinogram_out =  sinogram
        return sinogram_out


class CTTensorTransforms(Fanbeam2d):
    def __init__(
        self, 
        full_angles,
        img_shape, 
        mu_water=0.192,  #  0.192 [1/cm] = 192 [1/m]
        mu_air=0.0,
        source_distance=1075, 
        num_dets=672, 
        gpu_idx=0, 
        recon_mask=None, 
        det_distance=None, 
        det_spacing=None, 
        simul_poisson_rate=-1,
        simul_gaussian_rate=-1,
        min_hu=-1024,
        max_hu=3072
        ):
        super().__init__(img_shape, source_distance, num_dets, gpu_idx, recon_mask, det_distance, det_spacing)
        self.mu_water = mu_water
        self.mu_air = mu_air
        self.full_angles = np.array(full_angles)
        self.simul_poisson_rate = simul_poisson_rate
        self.simul_gaussian_rate = simul_gaussian_rate
        self.min_hu = min_hu
        self.max_hu = max_hu
    
    @staticmethod
    def equidist_sampling(elems, num_samples, sampling_axis=0):
        num_elems = np.array(elems).shape[sampling_axis]
        indices = np.linspace(0, num_elems-1, num_samples, False)
        indices = np.floor(indices).astype(int)
        if sampling_axis == 0:
            return elems[indices]
        elif sampling_axis == 1:
            return elems[:, indices]
        elif sampling_axis == 2:
            return elems[:, :, indices]
        else:
            raise NotImplementedError("Currently not implemented with sampling_axis outside of (0,1,2).")
    
    def get_indices_from_full_angles(self, ranges):
        """Get the corresponding indices of `angles` as a subset of `full_angles`."""
        # `ranges` is a list/tuple containing ranges of the selected views
        # For example, ranges = [(0, np.pi/6), (np.pi/3, np.pi/2)]
        # selects items from `self.full_angles` satisfying 
        #   0 <= items <= pi/6  or  pi/3 <= items <= pi/2
        selected_indices = []
        full_angles = self.full_angles
        for (min_val, max_val) in ranges:
            s = np.where(np.logical_and(full_angles >= min_val, full_angles <= max_val))[0]
            selected_indices.extend(s)
        return sorted(selected_indices)
    
    def low_dose_transform(self, sinogram, poisson_rate, gaussian_rate):
        sinogram = self.add_poisson_noise_to_sinogram(sinogram, noise_rate=poisson_rate)
        sinogram = self.add_gaussian_noise_to_sinogram(sinogram, sigma=gaussian_rate)
        return sinogram
    
    def sparse_view_transform(self, sinogram, num_views):
        # sinogram here is of the shape [#views, #dets]
        return self.equidist_sampling(sinogram, num_views, sampling_axis=0)
    
    def limited_angle_transform(self, sinogram, ranges):
        indices = self.get_indices_from_full_angles(ranges)
        return sinogram[indices]
    
    ### Standard preprocessing functions ###
    def get_simulated_sinogram_from_hu(self, hu_img, on_cuda=False):
        hu_img = self.clip_range(hu_img, input_hu=True, min_val=self.min_hu, max_val=self.max_hu)
        mu_img = np.float32(self.hu2mu(hu_img))
        sinogram = self.fp(mu_img, self.full_angles, use_gpu=True)
        
        sinogram = torch.from_numpy(sinogram).float().cuda() if on_cuda else sinogram
        sinogram = self.add_poisson_noise_to_sinogram(sinogram, self.simul_poisson_rate)
        sinogram = self.add_gaussian_noise_to_sinogram(sinogram, self.simul_gaussian_rate)
        return sinogram
    
    def get_simulated_image_from_hu(self, hu_img, on_cuda=False, **kwargs):
        sinogram = self.get_simulated_sinogram_from_hu(hu_img, on_cuda=on_cuda)
        if on_cuda:
            sinogram = sinogram.detach().cpu().numpy()
        else:
            sinogram = sinogram.numpy()
        image = self.fbp(sinogram, self.full_angles, use_gpu=True)
        return image
    
    
    def hu2mu(self, hu_img):
        mu_img = hu_img / 1000 * (self.mu_water - self.mu_air) + self.mu_water
        return mu_img

    def mu2hu(self, mu_img):
        hu_img = (mu_img - self.mu_water) / (self.mu_water - self.mu_air) * 1000
        return hu_img

    def clip_range(self, img, input_hu=True, min_val=-1024, max_val=3072):
        assert min_val < max_val
        img = self.mu2hu(img) if not input_hu else img
        img = img.clip(min_val, max_val)
        img = self.hu2mu(img) if not input_hu else img
        return img
    
    def get_windowed_hu(self, hu_img, width=3000, center=500, norm_to_255=False):
        ''' HU_img -> 0-1 normalization'''
        window_min = float(center) - 0.5 * float(width)
        win_img = (hu_img - window_min) / float(width)
        win_img[win_img < 0] = 0
        win_img[win_img > 1] = 1
        if norm_to_255:
            win_img = (win_img * 255).astype('float')
        return win_img

    def get_unwindowed_hu(self, win_img, width=3000, center=500, norm=False):
        ''' 0-1 normalization -> HU_img'''
        window_min = float(center) - 0.5 * float(width)
        if norm:
            win_img = win_img / 255.
        hu_img = win_img * float(width) + window_min
        return hu_img
    
    ### Noise simulation functions ###
    @staticmethod
    def add_poisson_noise_to_sinogram(sinogram, noise_rate=1e6):
        # noise rate: background intensity or source influx
        if not torch.is_tensor(sinogram):
            #print(f'Input should be tensor, got {type(sinogram)} (auto-converted).')
            if isinstance(sinogram, np.ndarray):
                sinogram = torch.from_numpy(sinogram)
            else:
                sinogram = torch.tensor(sinogram)
            
        if noise_rate > 0:
            max_val = sinogram.max() * 10 / 67  # this factor makes the noise more realistic?
            sinogram_ct = noise_rate * torch.exp(-sinogram / max_val)
            # add poison noise
            sinogram_noise = torch.poisson(sinogram_ct).clamp(min=sinogram_ct.min())
            sinogram_out = - max_val * torch.log(sinogram_noise / noise_rate)
        else:
            sinogram_out = sinogram
        return sinogram_out
    
    @staticmethod
    def add_gaussian_noise_to_sinogram(sinogram, sigma=0.1):
        if not torch.is_tensor(sinogram):
            #print(f'Input should be tensor, got {type(sinogram)} (auto-converted).')
            if isinstance(sinogram, np.ndarray):
                sinogram = torch.from_numpy(sinogram)
            else:
                sinogram = torch.tensor(sinogram)
        
        if sigma > 0:
            dtype = sinogram.dtype
            if not sinogram.is_floating_point():
                sinogram = sinogram.to(torch.float32)    
            sinogram_out = sinogram + sigma * torch.randn_like(sinogram)
            sinogram_out = sinogram_out.clamp(sinogram.min(), sinogram.max())
            if sinogram_out.dtype != dtype:
                sinogram_out = sinogram_out.to(dtype)
        else:
            sinogram_out =  sinogram
        return sinogram_out


if __name__ == '__main__':
    import SimpleITK as sitk
    import time
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    img_path = '/home/hejunjun/mcl_workspace/lowlevel/bin/L067_full/L067_FD_1_1.CT.0001.0001.2015.12.22.18.09.40.840353.358074219.dcm'
    img = sitk.GetArrayFromImage(sitk.ReadImage(img_path)).squeeze()
    full_angles = np.linspace(0, np.pi*2, 360)
    img_shape = img.shape[-2:]
    
    # time_list = []    
    # for _ in range(num_test):
    #     t0 = time.time()
    #     sino = ct_tool_tensor.get_simulated_image_from_hu(img, on_cuda=True)
    #     t1 = time.time()
    #     time_list.append(t1 - t0)
    # print('just for warm-up:', np.mean(time_list), sino.dtype)
    
    def test_run(use_tensor, use_numpy, on_cuda, num_test=20):
        time_list = []
        if use_tensor:
            ct_tool = CTTensorTransforms(full_angles=full_angles, img_shape=img_shape)
        else:
            ct_tool = CTTransforms(full_angles=full_angles, img_shape=img_shape)
            
        for _ in range(num_test):
            t0 = time.time()
            sino = ct_tool.get_simulated_image_from_hu(img, use_numpy=use_numpy, on_cuda=on_cuda)
            t1 = time.time()
            time_list.append(t1 - t0)
        print(f'use_tensor={use_tensor}, use_numpy={use_numpy}, on_cuda={on_cuda}: ', np.mean(time_list[len(time_list)//3:]))
    
    
    num_test = 10000
    #test_run(use_tensor=False, use_numpy=True, on_cuda=False, num_test=num_test)
    #test_run(use_tensor=False, use_numpy=True, on_cuda=True, num_test=num_test)
    #test_run(use_tensor=True, use_numpy=True, on_cuda=True, num_test=num_test)
    test_run(use_tensor=True, use_numpy=True, on_cuda=False, num_test=num_test)
    
    # seems use_tensor=True, on_cuda=False will be faster?
    
    
    
    

# class CTTool:
#     '''
#         1. mu-HU conversion
#         2. commonly used window conversion
#     '''
#     def __init__(self, mu_water=0.192):
#         self.mu_water = mu_water

#     def HU2mu(self, HU_img):
#         mu_img = HU_img / 1000 * self.mu_water + self.mu_water  # assuming MU_AIR = 0
#         return mu_img

#     def mu2HU(self, mu_img):
#         HU_img = (mu_img - self.mu_water) / self.mu_water * 1000
#         return HU_img

#     def clip_HU_range(self, img, input_HU=True, min_val=-1000, max_val=2000):
#         assert min_val < max_val
#         img = self.mu2HU(img) if not input_HU else img  # 如果传入的是mu,先转成HU
#         img[img < min_val] = min_val
#         img[img > max_val] = max_val
#         img = self.HU2mu(img) if not input_HU else img
#         return img
    
#     def window_transform(self, HU_img, width=3000, center=500, norm=False):
#         ''' HU_img -> 0-1 normalization'''
#         window_min = float(center) - 0.5 * float(width)
#         win_img = (HU_img - window_min) / float(width)
#         win_img[win_img < 0] = 0
#         win_img[win_img > 1] = 1
#         if norm:
#             print('normalized to 0-255')
#             win_img = (win_img * 255).astype('float')
#         return win_img

#     def back_window_transform(self, win_img, width=3000, center=500, norm=False):
#         ''' 0-1 normalization -> HU_img'''
#         window_min = float(center) - 0.5 * float(width)
#         if norm:
#             win_img = win_img / 255.
#         HU_img = win_img * float(width) + window_min
#         return HU_img


# # ==== common CT plot func ====
# def ct_with_mask(ct_image, mask, rgb_dict:dict = {'r':255,'g':0,'b':0}, mask_mode = None):
#     ''' plot CT with a color mask
#     args:
#         ct_image: 0-1 ct image
#         mask: mask, same shape as ct_image
#         rgb_dict: defalut red, 
#         mask_mode: plot with a larger mask, 'pool'-enlarge by pool2d, 'cycle'-enlarge by cycle
#     '''
#     # mask in (width, height)
#     if len(ct_image.size()) == 3:
#         ct_image = ct_image.squeeze(0)
#     elif len(ct_image.size()) == 4:
#         ct_image = ct_image.squeeze()

#     if len(mask.size()) > 2:
#         mask = mask.squeeze()

#     # 判断mask mode是否完整
#     if mask_mode is not None:
#         assert mask_mode in ['pool', 'cycle']
#         if mask_mode == 'pool':
#             mask = larger_mask_pool2d(mask)
#         else:
#             mask = larger_mask_cycle(mask)
    
#     ct_image = ct_image.unsqueeze(0)
#     ct_image_t = torch.repeat_interleave(ct_image, 3, dim=0)  # gray to rgb
#     ct_t_with_mask = ct_image_t.clone()
#     ct_t_with_mask[0,:][mask==1] = rgb_dict['r']
#     ct_t_with_mask[1,:][mask==1] = rgb_dict['g']
#     ct_t_with_mask[2,:][mask==1] = rgb_dict['b']
    
#     return ct_t_with_mask

# #=========enlarge the mask=======
# def larger_mask_cycle(mask, n=1,):
#     '''循环扩大mask[太慢]
#     n代表中心点范围,只要周围方块存在mask则填充
#     '''
#     count = 0 
#     width, height = mask.shape[:2]
#     new_mask = torch.zeros(width, height)
#     for i in range(width):
#         for j in range(height):
#             center_box = mask[max(0,i-n):min(width, i+n+1),max(0, j-n):min(height, j+n+1)]
#             if len(center_box[center_box==1])!= 0: # mask near the center
#                 count += 1
#                 new_mask[i][j] = 1 # new mask
#             else:
#                 new_mask[i][j] = 0
#     if isinstance(mask, np.ndarray):
#         # 只有ndarray需要扩展维度
#         new_mask = new_mask.unsqueeze(0).unsqueeze(1)
#     print(count)
#     return new_mask


# def larger_mask_pool2d(mask, n=3):
#     '''enlarge the mask by torch.maxpool2d
#     使用torch的maxpool2d实现膨胀操作
#     '''
#     k_size = n
#     assert k_size % 2 == 1 # k_size should be odd
#     mask_t = torch.from_numpy(mask) if isinstance(mask, np.ndarray) else mask
#     # reshape from (w,h)
#     mask_t = mask_t.unsqueeze(0) if len(mask_t.size()) < 3 else mask_t
        
#     # define the operator
#     pool = torch.nn.MaxPool2d(kernel_size=k_size, stride=1)
#     pad = torch.nn.ZeroPad2d(k_size//2)
#     # pad and pool
#     mask_t = pad(mask_t)
#     big_mask_t = pool(mask_t)
#     big_mask = big_mask_t.squeeze().numpy() if isinstance(mask, np.ndarray) else big_mask_t
#     return big_mask

# def save_ct(ct_image, path, **kwargs):
#     torchvision.utils.save_image(ct_image, path, **kwargs)


# def ct_diff(ct1, ct2, threshold=0.001):
#     '''calc ct difference by threshold
#     output:
#         diff: abs difference matrix
#         diff_mask: 0-1 mask by threshold
#     '''
#     assert ct1.size == ct2.size
#     diff = abs(ct1 - ct2)
#     diff_mask = torch.ones_like(ct1)  # one: white, mask用黑色表示
#     diff_mask[diff > threshold] = 0  # solid mask
#     return diff, diff_mask

