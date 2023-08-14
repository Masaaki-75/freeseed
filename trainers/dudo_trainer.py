import os
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from torch.utils.data import DataLoader
 
import sys
sys.path.append('..')
from trainers.basic_trainer import BasicTrainer
from datasets.aapmmyo import AAPMMyoDataset


class DuDoFreeNetTrainer(BasicTrainer):
    def __init__(self, opt=None, net=None, loss_type='l2'):
        super().__init__()
        assert opt is not None and net is not None
        self.num_views = opt.num_views
        self.net = net
        self.opt = opt
        self.criterion = self.get_pixel_criterion(loss_type)
        
        dataset_name = opt.dataset_name.lower()
        if dataset_name == 'aapm':
            self.train_dataset = AAPMMyoDataset(opt.dataset_path, mode='train', dataset_shape=opt.dataset_shape)
            self.val_dataset = AAPMMyoDataset(opt.dataset_path, mode='val', dataset_shape=opt.dataset_shape)
        else:
            raise NotImplementedError(f'Dataset {dataset_name} not implemented, try aapm.')
        
        # save path
        self.checkpoint_path = os.path.join(opt.checkpoint_root, opt.checkpoint_dir)
        if opt.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(os.path.join(opt.tensorboard_root, opt.tensorboard_dir))
        if opt.use_wandb:
            if opt.local_rank == 0:  # only on main process
                self.wandb_init(self.opt)
        self.itlog_intv = opt.log_interval

    def generate_sparse_and_gt_data(self, mu_ct, return_sinomask=False, mixed_interp=False):
        if return_sinomask:
            try:
                sparse_sinogram, sparse_mu, sinogram_full, gt_mu, sino_mask = self.net.generate_sparse_and_full_dudo(mu_ct, return_sinomask=return_sinomask, mixed_interp=mixed_interp)
            except nn.modules.module.ModuleAttributeError:
                sparse_sinogram, sparse_mu, sinogram_full, gt_mu, sino_mask = self.net.module.generate_sparse_and_full_dudo(mu_ct, return_sinomask=return_sinomask, mixed_interp=mixed_interp)
            return sparse_sinogram, sparse_mu, sinogram_full, gt_mu, sino_mask
        else:
            try:
                sparse_sinogram, sparse_mu, sinogram_full, gt_mu = self.net.generate_sparse_and_full_dudo(mu_ct, mixed_interp=mixed_interp)
            except nn.modules.module.ModuleAttributeError:
                sparse_sinogram, sparse_mu, sinogram_full, gt_mu = self.net.module.generate_sparse_and_full_dudo(mu_ct, mixed_interp=mixed_interp)
            return sparse_sinogram, sparse_mu, sinogram_full, gt_mu

    # reconstruct_trainer fit
    def fit(self,):
        opt = self.opt
        self.net = nn.SyncBatchNorm.convert_sync_batchnorm(self.net)
        torch.cuda.set_device(opt.local_rank)
        dist.init_process_group(backend='nccl')
        device = torch.device('cuda', opt.local_rank)

        print(f'''Summary:
            Sparse Angles:         {opt.num_views}
            Number of Epochs:      {opt.epochs}
            Batch Size:            {opt.batch_size}
            Initial Learning rate: {opt.lr}
            Training Size:         {len(self.train_dataset)}
            Validation Size:       {len(self.val_dataset)}
            Checkpoints Saved:     {opt.checkpoint_dir}
            Checkpoints Loaded:    {opt.net_checkpath}
            Device:                {device}
        ''')

        # resume model
        if self.opt.resume:
            self.resume()
        else:
            try:
                self.weights_init(self.net)
            except Exception as err:
                print(f'init failed: {err}')
                
        # network to device, DDP
        self.net = self.net.to(device)
        self.net = torch.nn.parallel.DistributedDataParallel(
            self.net, device_ids=[opt.local_rank], output_device=opt.local_rank, find_unused_parameters=True)
        train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset)
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, sampler=train_sampler, pin_memory=True,)
        val_sampler = torch.utils.data.distributed.DistributedSampler(self.val_dataset)
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=1, num_workers=opt.num_workers, sampler=val_sampler,)

        # init and resume optimizer
        self.optimizer = torch.optim.Adam(self.net.img_net.parameters(), lr=opt.lr)  # for full training
        self.optimizer2 = torch.optim.Adam(self.net.parameters(), lr=opt.pretrain_lr)  # for FreeNet pretraining
        self.scheduler = self.get_scheduler(self.optimizer)

        # start pretraining
        for epoch in range(opt.pretrain_epochs):
            print(f'start pre-training epoch: {epoch}')
            self.train_loader.sampler.set_epoch(epoch)
            self.pretrain()
        
        # start training
        start_epoch = self.epoch
        self.iter = 0
        for self.epoch in range(start_epoch, opt.epochs):
            print(f'start training epoch: {self.epoch}')
            self.train_loader.sampler.set_epoch(self.epoch)
            self.train()
            self.val()
            self.scheduler.step()
            save_condition = ((self.epoch + 1) % self.opt.save_epochs == 0) or ((self.epoch+1) == self.opt.epochs)
            if self.opt.local_rank == 0 and save_condition:
                self.save_model()
                #self.save_opt()

    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    
    def pretrain(self):
        self.net = self.net.train()
        self.set_requires_grad(self.net.sino_net, False)  # freeze sinogram domain network first
        pbar = tqdm.tqdm(self.train_loader, ncols=60) if self.opt.use_tqdm else self.val_loader
        for i, data in enumerate(pbar):  # one batch contain different images
            mu_ct = data.to('cuda')
            
            sparse_sinogram, sparse_mu, _, gt_mu, sino_mask = self.generate_sparse_and_gt_data(mu_ct, return_sinomask=True, mixed_interp=False)
            output_mu, _, output_sino_img = self.net(sparse_sinogram, sino_mask, sparse_mu, only_train_img_net=True)
            loss = self.criterion(output_mu, gt_mu) + self.criterion(output_sino_img, gt_mu)
            
            self.optimizer2.zero_grad()
            loss.backward()
            self.optimizer2.step()
                
            if self.opt.use_tqdm:
                pbar.set_postfix({'loss': '%.2f' % (loss.item())})
                pbar.update(1)
    
    
    def train(self,):
        self.iter_log_flag = False
        losses, rmses, psnrs, ssims = [], [], [], []

        # train the model
        self.net = self.net.train()
        self.set_requires_grad(self.net.sino_net, True)
        pbar = tqdm.tqdm(self.train_loader, ncols=60) if self.opt.use_tqdm else self.val_loader
        for i, data in enumerate(pbar):
            mu_ct = data.to('cuda')
            sparse_sinogram, sparse_mu, gt_sinogarm, gt_mu, sino_mask = self.generate_sparse_and_gt_data(mu_ct, return_sinomask=True, mixed_interp=False)
            output_mu, ouput_sinogram, output_sino_img = self.net(sparse_sinogram, sino_mask, sparse_mu, only_train_img_net=False)
            loss = self.criterion(output_mu, gt_mu) + self.criterion(output_sino_img, gt_mu) + self.criterion(ouput_sinogram, gt_sinogarm)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            losses.append(loss.item())
            rmse, psnr, ssim = self.get_metrics_by_window(output_mu, gt_mu)
            rmses.append(rmse)
            psnrs.append(psnr)
            ssims.append(ssim)

            # log acc by iteration
            if self.opt.local_rank == 0:
                if self.iter !=0 and self.iter % self.itlog_intv == 0:
                    log_info = {
                        'loss': np.mean(losses[-self.itlog_intv:]),
                        'rmse': np.mean(rmses[-self.itlog_intv:]),
                        'ssim': np.mean(ssims[-self.itlog_intv:]),
                        'psnr': np.mean(psnrs[-self.itlog_intv:])}
                    if self.opt.use_tensorboard:
                        self.tensorboard_scalar(self.writer, 'train/loss', self.iter, **log_info)
                    if self.opt.use_wandb:
                        self.wandb_logger('train/iter' ,**log_info)
                self.iter += 1

        # epoch info
        if self.opt.local_rank == 0:
            epoch_log = {
            'loss': np.mean(losses),
            'rmse': np.mean(rmses),
            'ssim': np.mean(ssims),
            'psnr': np.mean(psnrs),}

            current_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            
            print(f'Epoch {self.epoch} learning rate: {current_lr}')
            print(f'Epoch {self.epoch} train loss: {epoch_log["loss"]}')
            print(f'Epoch {self.epoch} train rmse: {epoch_log["rmse"]}')
            print(f'Epoch {self.epoch} train ssim: {epoch_log["ssim"]}')
            print(f'Epoch {self.epoch} train psnr: {epoch_log["psnr"]}')
            
            if self.opt.use_tensorboard:
                self.tensorboard_scalar(self.writer, 'train/epoch', self.epoch, **epoch_log)
                self.tensorboard_scalar(self.writer, 'settings', self.epoch, **{'current_lr':current_lr,'batch_size':self.opt.batch_size})
            if self.opt.use_wandb:
                self.wandb_logger('train/epoch', step_name='epoch', step=self.epoch, **epoch_log)
                self.wandb_logger('settings', step_name='epoch', step=self.epoch, **{'current_lr':current_lr,'batch_size':self.opt.batch_size})

    def val(self,):
        self.iter_log_flag = False
        print(f'start validation epoch: {self.epoch}')
        losses, rmses, psnrs, ssims = [], [], [], []
        self.net = self.net.eval()
        pbar = tqdm.tqdm(self.val_loader, ncols=60) if self.opt.use_tqdm else self.val_loader
        with torch.no_grad():
            for i, data in enumerate(pbar):
                mu_ct = data.to('cuda')
                sparse_sinogram, sparse_mu, gt_sinogarm, gt_mu, sino_mask = self.generate_sparse_and_gt_data(mu_ct, return_sinomask=True, mixed_interp=False)
                output_mu, ouput_sinogram, output_sino_img = self.net(sparse_sinogram, sino_mask, sparse_mu, only_train_img_net=False)
                loss = self.criterion(output_mu, gt_mu) + self.criterion(output_sino_img, gt_mu) + self.criterion(ouput_sinogram, gt_sinogarm)
                
                losses.append(loss.item())
                rmse, psnr, ssim = self.get_metrics_by_window(output_mu, gt_mu)
                rmses.append(rmse)
                psnrs.append(psnr)
                ssims.append(ssim)
            
        if self.opt.local_rank == 0:
            print('Logging validation information...')
            epoch_log = {
                'loss': np.mean(losses),
                'rmse': np.mean(rmses),
                'ssim': np.mean(ssims),
                'psnr': np.mean(psnrs),}
            
            print(f'Epoch {self.epoch} val loss: {epoch_log["loss"]}')
            print(f'Epoch {self.epoch} val rmse: {epoch_log["rmse"]}')
            print(f'Epoch {self.epoch} val ssim: {epoch_log["ssim"]}')
            print(f'Epoch {self.epoch} val psnr: {epoch_log["psnr"]}')
            
            if self.opt.use_tensorboard:
                self.tensorboard_scalar(self.writer, 'val/epoch', self.epoch, **epoch_log)
            if self.opt.use_wandb:
                self.wandb_logger('val/epoch', step_name='epoch', step=self.epoch, **epoch_log)
