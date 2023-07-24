import os
import tqdm
import random
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
 
import sys
sys.path.append('..')
from trainers.basic_trainer import BasicTrainer
from datasets.aapmmyo import AAPMMyoDataset, CTTools
from utilities.metrics import compute_measure


class SimpleTester(BasicTrainer):
    def __init__(self, opt, net, test_window=None, data_range=1, **kwargs):
        super().__init__()
        self.opt = opt
        self.net = net
        self.net2 = kwargs['net2'] if 'net2' in kwargs.keys() else None
        self.num_views = self.opt.num_views
        self.cttool = CTTools()
        if test_window is not None:
            assert isinstance(test_window, list)
            print('Test window: ', test_window)
        else:
            test_window = [(3000,500),(800,-600),(500,50),(2000,0)]
        self.test_window = test_window
        self.save_fig = self.opt.tester_save_image
        self.data_range = data_range
        self.tables = defaultdict(dict)  # simply record metrics
        self.tables_stat = defaultdict(dict)  # record statistics
        for (width, center) in self.test_window:
            for metric_name in ['psnr', 'ssim', 'rmse']:
                self.tables[f'({width},{center})_' + metric_name] = []  # initialize multiple tables, each containing a list
                #self.tables_stat[f'({width},{center})_' + metric_name] = []
        
        self.save_dir = os.path.join(self.opt.tester_save_path, self.opt.tester_save_name)
        os.makedirs(self.save_dir, exist_ok=True)
        print(f'Save figures to {self.save_dir}? : ', self.save_fig)
        self.saved_slice = 0
        self.seed_torch(seed=1)
    
    def seed_torch(self, seed=1):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    def prepare_dataset(self,):
        opt = self.opt
        dataset_name = opt.dataset_name.lower()
        if dataset_name == 'aapm':
            self.test_dataset = AAPMMyoDataset(opt.dataset_path, mode='test', dataset_shape=opt.dataset_shape)
        else:
            raise NotImplementedError(f'Dataset {dataset_name} not implemented, try aapm.')
        self.test_loader = DataLoader(self.test_dataset, batch_size=1, num_workers=1,)

    def model_forward(self, data):
        if self.opt.network in ['freenet', 'seednet']:
            sparse_mu, gt_mu = self.net.generate_sparse_and_full_ct(data)
            output_mu = self.net(sparse_mu)
            test_dict = {'output_ct': output_mu,}
            return sparse_mu, gt_mu, test_dict

        elif self.opt.network in ['fbp']:
            sparse_mu, gt_mu = self.net.generate_sparse_and_full_ct(data)
            test_dict = {'sparse_ct': sparse_mu}
            return sparse_mu, gt_mu, test_dict

        elif self.opt.network in ['dudonet', 'dudofreenet']:
            sparse_sinogram, sparse_mu, _, gt_mu, sino_mask = self.net.generate_sparse_and_full_dudo(data, return_sinomask=True)
            output_mu, _, output_sino_img = self.net(sparse_sinogram, sino_mask, sparse_mu,)
            test_dict = {'output_ct': output_mu,}
            return sparse_mu, gt_mu, test_dict
        
        elif 'freeseed' in self.opt.network:
            #assert self.net2 is not None, 'SeedNet required but not provided.'
            sparse_mu, gt_mu = self.net.generate_sparse_and_full_ct(data)
            neg_art, output_mu = self.net(sparse_mu)
            # output_mu = neg_art + sparse_mu
            # proxy_mu = self.net2(output_mu)
            # mask = rescale_tensor(-neg_art.detach())
            # mask = threshold_tensor(mask)
            test_dict = {
                'output_ct': output_mu,
                #'proxy_ct': proxy_mu,
            }
            return sparse_mu, gt_mu, test_dict
        else:
            raise NotImplementedError(f'network {self.opt.network} not implemented')

    def run(self,):
        self.prepare_dataset()

        # a simple tester
        if 'freeseed' in self.opt.network:
            #assert self.net2 is not None
            self.net = self.load_model(net=self.net, net_checkpath=self.opt.net_checkpath, output=True)
            if self.opt.net_checkpath2:
                self.net2 = self.load_model(net=self.net2, net_checkpath=self.opt.net_checkpath2, output=True)
        elif self.opt.network != 'fbp':
            self.load_model()
        
        self.net = self.net.cuda()
        self.net = self.net.eval()
        if self.net2 is not None:
            self.net2 = self.net2.cuda()
            self.net2 = self.net2.eval()

        pbar = tqdm.tqdm(self.test_loader, ncols=60)
        with torch.no_grad():
            for i, data in enumerate(pbar):
                data = data.cuda()
                sparse_mu, gt_mu, test_dict = self.model_forward(data)
                self.test_batch(sparse_mu, gt_mu, **test_dict)
        self.save_csv()

    def test_batch(self, sparse_mu, gt_mu, **kwargs):
        assert len(kwargs) > 0
        key_list = [k for k in kwargs.keys()]
        batch = kwargs[key_list[0]].shape[0]
        for b in range(batch):
            single_kwargs = {}
            for key in key_list:
                single_kwargs[key] = kwargs[key][b:b+1]
            self.test_slice(sparse_mu[b:b+1], gt_mu[b:b+1], **single_kwargs)

    def test_slice(self, sparse_mu, gt_mu, **kwargs):
        gt_hu = self.cttool.mu2HU(gt_mu)
        sparse_hu = self.cttool.mu2HU(sparse_mu)
        for (width, center) in self.test_window:
            for key, value in kwargs.items():
                value_hu = self.cttool.mu2HU(value)
                value_win = self.cttool.window_transform(value_hu, width=width, center=center)
                gt_win = self.cttool.window_transform(gt_hu, width=width, center=center)
                #sparse_win = self.cttool.window_transform(sparse_hu, width=width, center=center)
                rmse, psnr, ssim = compute_measure(value_win, gt_win, self.data_range)
                self.tables[f'({width},{center})_psnr'].append(psnr)
                self.tables[f'({width},{center})_ssim'].append(ssim)
                self.tables[f'({width},{center})_rmse'].append(rmse)

                if self.save_fig:
                    self.save_png(value_win, f'{key}', window_name=f'({width},{center})')

            # if self.save_fig:
            #     self.save_png(sparse_win, 'aa', window_name=f'({width},{center})')
            #     self.save_png(gt_win, 'gt', window_name=f'({width},{center})')
        self.saved_slice += 1

    def save_png(self, value, name, window_name):
        save_dir = os.path.join(self.save_dir, window_name)
        os.makedirs(save_dir, exist_ok=True)
        saved_slice = str(self.saved_slice).rjust(3, '0')
        if name in ['gt', 'aa', 'input']:
            fullname = f'{saved_slice}_{name}.png'
        else:
            fullname = f'{saved_slice}_{name}_{self.opt.network}.png'
        save_path = os.path.join(save_dir,fullname)
        save_image(value, save_path, normalize=False)
    
    def write_stat_table(self,):
        for (width, center) in self.test_window:
            for metric_name in ['psnr', 'ssim', 'rmse']:
                table_tmp = self.tables[f'({width},{center})_' + metric_name]
                self.tables_stat[f'({width},{center})']['avg_' + metric_name] = np.mean(table_tmp)
                self.tables_stat[f'({width},{center})']['std_' + metric_name] = np.std(table_tmp)
                self.tables_stat[f'({width},{center})']['min_' + metric_name] = np.min(table_tmp)
                self.tables_stat[f'({width},{center})']['max_' + metric_name] = np.max(table_tmp)
            print(f"Averaged PSNR under window {(width, center)}: {self.tables_stat[f'({width},{center})']['avg_psnr']}")
            print(f"Averaged SSIM under window {(width, center)}: {self.tables_stat[f'({width},{center})']['avg_ssim']}")
            print(f"Averaged RMSE under window {(width, center)}: {self.tables_stat[f'({width},{center})']['avg_rmse']}")

    def save_csv(self,):
        self.write_stat_table()
        df = pd.DataFrame(self.tables)
        csv_path = os.path.join(self.save_dir, self.opt.network + str(self.opt.num_views) +'_all.csv')
        df.to_csv(csv_path)
        print('Table written in: ', csv_path)

        df_stat = pd.DataFrame(self.tables_stat)
        csv_stat_path = os.path.join(self.save_dir, self.opt.network + str(self.opt.num_views) +'_stat.csv')
        df_stat.to_csv(csv_stat_path)
        print('Table (stat) written in: ', csv_stat_path)
        print(df_stat)