import os
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

import sys
sys.path.append('..')
from datasets.aapmmyo import CTTools
from utilities.metrics import compute_measure

class BasicTrainer:
    def __init__(self):
        self.iter = 0
        self.epoch = 0
        self.cttool = CTTools()
        
    @staticmethod
    def weights_init(m):             
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)    

    def get_optimizer(self, net):
        opt = self.opt
        optimizer_name = opt.optimizer
        assert isinstance(optimizer_name, str)
        optimizer_name = optimizer_name.lower()
        if optimizer_name == 'adam':
            return torch.optim.Adam(net.parameters(), lr=opt.lr, betas=(opt.beta1, self.opt.beta2))
        elif optimizer_name == 'sgd':
            return torch.optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
        elif optimizer_name == 'adamw':
            return torch.optim.AdamW(net.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
        elif optimizer_name == 'lbfgs':
            return torch.optim.LBFGS(net.parameters(), lr=opt.lr, tolerance_grad=-1, tolerance_change=-1)
        else:
            raise NotImplementedError(f'Currently only support optimizers among Adam/AdamW/SGD/LBFGS, got {optimizer_name}.')
    
    def get_scheduler(self, optimizer):
        opt = self.opt
        scheduler_name = opt.scheduler
        assert isinstance(scheduler_name, str)
        scheduler_name = scheduler_name.lower()
        if scheduler_name == 'step':
            return optim.lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=opt.step_gamma)
        elif scheduler_name == 'mstep':
            return optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.step_gamma)
        elif scheduler_name == 'exp':
            return optim.lr_scheduler.ExponentialLR(optimizer, gamma=opt.step_gamma)
        elif scheduler_name == 'poly':
            return optim.lr_scheduler.PolynomialLR(optimizer, total_iters=opt.poly_iters, power=opt.poly_power)
        elif scheduler_name == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min=1e-6)
        else:
            raise NotImplementedError(f'Currently only support schedulers among Step/MultiStep/Exp/Poly/Cosine, got {scheduler_name}.')

    def reduce_value(self, value, average=True):
        world_size = dist.get_world_size()
        if world_size < 2:  # single GPU
            return value
        if not torch.is_tensor(value):
            value = torch.tensor(value)
        if not value.is_cuda:
            value = value.cuda(self.opt.local_rank)
        with torch.no_grad():
            dist.all_reduce(value)   # get reduce value
            if average:
                value = value.float()
                value /= world_size
        return value.cpu()

    @staticmethod
    def save_checkpoint(param, path, name:str, epoch:int):
        # simply save the checkpoint by epoch
        if not os.path.exists(path):
            os.makedirs(path)
        checkpoint_path = os.path.join(path, name + '_{}_epoch.pkl'.format(epoch))
        torch.save(param, checkpoint_path)

    def save_model(self, net=None, net_name=''):
        net_param = net.module.state_dict() if net is not None else self.net.module.state_dict() # default multicard
        checkpoint_path = os.path.join(self.opt.checkpoint_root, self.opt.checkpoint_dir)
        net_check = {'net_param': net_param, 'epoch': self.epoch,}
        self.save_checkpoint(net_check, checkpoint_path, self.opt.checkpoint_dir + '-net-' + net_name, self.epoch)

    def save_opt(self, optimizer=None, scheduler=None, opt_name=''):
        checkpoint_path = os.path.join(self.opt.checkpoint_root, self.opt.checkpoint_dir)
        optimizer_param = optimizer.state_dict() if optimizer is not None else self.optimizer.state_dict()
        scheduler_param = scheduler.state_dict() if scheduler is not None else self.scheduler.state_dict()
        opt_check = {
            'optimizer': optimizer_param,
            'scheduler': scheduler_param,
            'epoch' : self.epoch,
        }
        self.save_checkpoint(opt_check, checkpoint_path, self.opt.checkpoint_dir +'-opt-' + opt_name, self.epoch)

    def load_model(self, net=None, net_checkpath=None, output=False):
        net_checkpath = self.opt.net_checkpath if net_checkpath is None else net_checkpath
        net_checkpoint = torch.load(net_checkpath, map_location='cpu')
        net_checkpoint = net_checkpoint['net_param'] if 'net_param' in net_checkpoint.keys() else net_checkpoint
        if net is None:
            self.net.load_state_dict(net_checkpoint, strict=True)
        else:
            net.load_state_dict(net_checkpoint, strict=False)
        print('finish loading network')
        if output:
            return net

    def load_opt(self):
        opt_checkpath = self.opt.opt_checkpath
        opt_checkpoint = torch.load(opt_checkpath, map_location='cpu')
        self.optimizer.load_state_dict(opt_checkpoint['optimizer'])
        self.scheduler.load_state_dict(opt_checkpoint['scheduler'])
        self.epoch = opt_checkpoint['epoch']
        print('finish loading opt')

    def resume(self, net=None):
        if self.opt.net_checkpath is not None:
            self.load_model()
        else:
            raise ValueError('opt.net_checkpath not provided.')

    def resume_opt(self,):
        if self.opt.resume_opt and self.opt.opt_checkpath is not None:
            self.load_opt()
            print('finish loading optimizer')
        else:
            print('opt.opt_checkpath not provided')     

    def get_pixel_criterion(self, mode=None, reduction='mean'):
        mode = self.opt.loss_type if mode is None else mode
        assert isinstance(mode, str)
        mode = mode.lower()
        if mode == 'l1':
            criterion = torch.nn.L1Loss(reduction=reduction) 
        elif mode == 'sml1':
            criterion = torch.nn.SmoothL1Loss(reduction=reduction)
        elif mode == 'l2':
            criterion = torch.nn.MSELoss(reduction=reduction)
        elif mode in ['crossentropy', 'ce', 'cross_entropy']:
            criterion = torch.nn.CrossEntropyLoss()
        elif mode in ['cos', 'consine']:
            def cosine_similarity_loss(x, y): 
                sim = torch.cosine_similarity(x, y, dim=0)
                if reduction == 'mean':
                    return 1. - sim.mean()
                else:
                    return 1. - sim.sum()
            criterion = cosine_similarity_loss
        else:
            raise NotImplementedError('pixel_loss error: mode not in [l1, sml1, l2, ce].')
        return criterion
    
    # ---- basic logging function ----
    @staticmethod
    def tensorboard_scalar(writer, rel_path, step, **kwargs):
        for key in kwargs.keys():
            path = os.path.join(rel_path, key)
            writer.add_scalar(path, kwargs[key], global_step=step)

    @staticmethod
    def tensorboard_image(writer, rel_path, **kwargs):
        for key in kwargs.keys():
            path = os.path.join(rel_path, key)
            writer.add_image(tag=path, img_tensor=kwargs[key], global_step=0, dataformats='CHW',)
    
    @staticmethod
    def wandb_init(opt, key=None):
        if key is None:
            print('WANDB key not provided, attempting anonymous login...')
        else:
            wandb.login(key=key)
        wandb_root = opt.tensorboard_root if opt.wandb_root == '' else opt.wandb_root
        wandb_dir = opt.tensorboard_dir if opt.wandb_dir == '' else opt.wandb_dir
        wandb_path = os.path.join(wandb_root, wandb_dir)
        if not os.path.exists(wandb_path):
            os.makedirs(wandb_path)
        wandb.init(project=opt.wandb_project, name=str(wandb_dir), dir=wandb_path, resume='allow', reinit=True,)
    
    @staticmethod
    def to_wandb_img(**kwargs):
        # turn torch makegrid to wandb image
        for key, value in kwargs.items():
            kwargs[key] = wandb.Image(kwargs[key].float().cpu())
        return kwargs

    @staticmethod
    def wandb_logger(r_path=None, step_name=None, step=None, **kwargs):
        log_info = {}
        if step is not None:
            log_info.update({str(step_name): step})
        for key, value in kwargs.items():
            if r_path is not None:
                key_name = str(os.path.join(r_path, key))
            else:
                key_name = key
            log_info[key_name] = kwargs[key]
        wandb.log(log_info)
        
    @staticmethod
    def wandb_scalar(r_path, step=None, **kwargs):
        for key in kwargs.keys():
            if step is not None:
                wandb.log({'{}'.format(os.path.join(r_path, key)): kwargs[key]}, step=step)
            else:
                wandb.log({'{}'.format(os.path.join(r_path, key)): kwargs[key]})
    
    @staticmethod
    def wandb_image(step=None, **kwargs):
        for key in kwargs.keys():
            kwargs[key] = wandb.Image(kwargs[key].float().cpu())
        if step is not None:
            wandb.log(kwargs, step=step)
        else:
            wandb.log(kwargs)

    # basic Sparse_CT accuracy by window
    def get_metrics_by_window(self, mu_input, mu_target, to_HU=True):
        # calculate mu ct accuracy by CT window
        if to_HU:
            hu_input = self.cttool.window_transform(self.cttool.mu2HU(mu_input))
            hu_target = self.cttool.window_transform(self.cttool.mu2HU(mu_target))
        else:
            hu_input, hu_target = mu_input, mu_target
        # data_range of normalized HU img
        data_range = 1
        rmse, psnr, ssim = compute_measure(hu_input, hu_target, data_range)
        return rmse, psnr, ssim
    
    # ---- training fucntion ----
    def fit(self):
        raise NotImplementedError('function fit() not implemented.')

    def train(self):
        raise NotImplementedError('function train() not implemented.')

    def val(self):
        raise NotImplementedError('function val() not implemented.')