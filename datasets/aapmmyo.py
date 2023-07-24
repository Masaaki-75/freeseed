import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class CTTools:
    def __init__(self, mu_water=0.192):
        self.mu_water = mu_water

    def HU2mu(self, hu_img):
        mu_img = hu_img / 1000 * self.mu_water + self.mu_water
        return mu_img

    def mu2HU(self, mu_img):
        hu_img = (mu_img - self.mu_water) / self.mu_water * 1000
        return hu_img

    def window_transform(self, hu_img, width=3000, center=500, norm=False):
        # HU -> 0-1 normalized 
        min_window = float(center) - 0.5 * float(width)
        win_img = (hu_img - min_window) / float(width)
        win_img[win_img < 0] = 0
        win_img[win_img > 1] = 1
        if norm:
            print('normalize to 0-255')
            win_img = (win_img * 255).astype('float')
        return win_img

    def back_window_transform(self, win_img, width=3000, center=500, norm=False):
        # 0-1 normalized -> HU
        min_window = float(center) - 0.5 * float(width)
        win_img = win_img / 255 if norm else win_img
        hu_img = win_img * float(width) + min_window
        return hu_img
    

class AAPMMyoDataset(Dataset):
    def __init__(self, src_path_list, dataset_shape=512, spatial_dims=2, mode='train', num_train=5410, num_val=526):
        assert mode in ['train', 'val'], f'Invalid mode: {mode}.'
        self.mode = mode
        self.num_train = num_train
        self.num_val = num_val
        self.cttool = CTTools()
        if not isinstance(dataset_shape, (list, tuple)):
            dataset_shape = (dataset_shape,) * spatial_dims
        self.dataset_shape = dataset_shape
        
        self.src_path_list = self.get_path_list_from_dir(src_path_list, ext='.npy', keyword='')
        print(f'finish loading AAPM-myo {mode} dataset, total images {len(self.src_path_list)}')

    def get_path_list_from_dir(self, src_dir, ext='.npy', keyword=''):
        assert isinstance(src_dir, (list, tuple)) or (isinstance(src_dir, str) and os.path.isdir(src_dir)), \
            f'Input should either be a directory containing taget files or a list containing paths of target files, got {src_dir}.'
        
        if isinstance(src_dir, str) and os.path.isdir(src_dir):
            src_path_list = sorted([os.path.join(src_dir, _) for _ in os.listdir(src_dir) if ext in _ and keyword in _])
        elif isinstance(src_dir, (list, tuple)):
            src_path_list = src_dir
        
        mode = self.mode
        train_path_list, val_path_list = self.simple_split(src_path_list, self.num_train, self.num_val)
        return train_path_list if mode == 'train' else val_path_list
    
    @staticmethod
    def simple_split(path_list, num_train=None, num_val=None):
        num_imgs = len(path_list)
        if num_train is None or num_val is None:
            num_train = int(num_imgs * 0.8)
            num_val = num_imgs - num_train
            
        if num_train > num_val:
            train_list = path_list[:num_train]
            val_list = path_list[-num_val:]
            print('dataset:{}, training set:{}, val set:{}'.format(len(path_list), len(train_list), len(val_list)))
        else:
            raise ValueError(f'aapm_myo dataset simple_split() error. num_imgs={num_imgs}, while num_train={num_train}, num_val={num_val}.')
        return train_list, val_list
    
    def __getitem__(self, idx):
        src_path = self.src_path_list[idx]
        src_hu = np.load(src_path).squeeze()
        W, H = src_hu.shape[-1], src_hu.shape[-2]

        if H != self.dataset_shape[0] or W != self.dataset_shape[1]:
            src_hu = cv2.resize(src_hu, self.dataset_shape, cv2.INTER_CUBIC)
            
        src_mu = self.cttool.HU2mu(src_hu)
        src_mu = torch.from_numpy(src_mu).unsqueeze(0).float()
        return src_mu

    def __len__(self):
        return len(self.src_path_list)




if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import tqdm  
    root_path = '/mnt/data_jixie1/clma/aapm_tr5410_te526'
    # image_list_path = './image_list.txt'
    aapm_dataset = AAPMMyoDataset(root_path, mode='val', dataset_shape=256, num_train=5410, num_val=526)
    val_loader = DataLoader(aapm_dataset, batch_size=1, num_workers=2)
    pbar = tqdm.tqdm(val_loader, ncols=60)
    for i, data in enumerate(pbar):
        print(i)
        print(data.shape)
        break