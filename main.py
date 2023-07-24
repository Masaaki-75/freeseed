import os
import sys
import random
import argparse
import numpy as np
import torch
import torch.nn as nn

from networks.freeseed import FreeNet, SeedNet
from networks.dudofree import DuDoFreeNet
from trainers.simple_trainer import SimpleTrainer
from trainers.freeseed_trainer import FreeSeedTrainer
from trainers.simple_tester import SimpleTester
from trainers.dudo_trainer import DuDoFreeNetTrainer


def get_parser():
    parser = argparse.ArgumentParser(description='Sparse CT Main')
    # logging interval by iteration
    parser.add_argument('--log_interval', type=int, default=400, help='logging interval by iteration')
    # tensorboard config
    parser.add_argument('--checkpoint_root', type=str, default='', help='where to save the checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, default='test', help='detail folder of checkpoint')
    parser.add_argument('--use_tensorboard', action='store_true', default=False, help='whether to use tensorboard')
    parser.add_argument('--tensorboard_root', type=str, default='', help='root path of tensorboard, project path')
    parser.add_argument('--tensorboard_dir', type=str, required=True, help='detail folder of tensorboard')
    # wandb config
    parser.add_argument('--use_tqdm', action='store_true', default=False, help='whether to use tqdm')
    parser.add_argument('--use_wandb', action='store_true', default=False, help='whether to use wandb')
    parser.add_argument('--wandb_project', type=str, default='Sparse_CT')
    parser.add_argument('--wandb_root', type=str, default='')
    parser.add_argument('--wandb_dir', type=str, default='')
    # DDP
    parser.add_argument('--local_rank', type=int, default=-1, help='node rank for torch distributed training')
    # data_path
    parser.add_argument('--dataset_path', type=str, default='', help='dataset path')
    parser.add_argument('--dataset_name', default='aapm', type=str, help='which dataset, size640,size320,deepleision.etc.')
    parser.add_argument('--dataset_shape', type=int, default=512, help='modify shape in dataset')
    parser.add_argument('--num_train', default=5410, type=int, help='number of training examples')
    parser.add_argument('--num_val', default=526, type=int, help='number of validation examples')
    # dataloader
    parser.add_argument('--batch_size', default=4, type=int, help='batch_size')    
    parser.add_argument('--shuffle', default=True, type=bool, help='dataloader shuffle, False if test and val')
    parser.add_argument('--num_workers', default=4, type=int, help='dataloader num_workers, 4 is a good choice')
    parser.add_argument('--drop_last', default=False, type=bool, help='dataloader droplast')
    # optimizer
    parser.add_argument('--optimizer', default='adam', type=str, help='name of the optimizer')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')    
    parser.add_argument('--beta1', default=0.5, type=float, help='Adam beta1')    
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam beta2')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum for SGD optimizer')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay for optimizer')
    parser.add_argument('--epochs', default=30, type=int, help='number of training epochs')
    parser.add_argument('--save_epochs', default=10, type=int)
    # scheduler
    parser.add_argument('--scheduler', default='step', type=str, help='name of the scheduler')
    parser.add_argument('--step_size', default=10, type=int, help='step size for StepLR')
    parser.add_argument('--milestones', nargs='+', type=int, help='milestones for MultiStepLR')
    parser.add_argument('--step_gamma', default=0.5, type=float, help='learning rate reduction factor')
    parser.add_argument('--poly_iters', default=10, type=int, help='the number of steps that the scheduler decays the learning rate')
    parser.add_argument('--poly_power', default=2, type=float, help='the power of the polynomial')
    
    # checkpath && resume training
    parser.add_argument('--resume', default=False, action='store_true', help='resume network training or not, load network param')
    parser.add_argument('--resume_opt', default=False, action='store_true', help='resume optimizer or not, load opt param')
    parser.add_argument('--net_checkpath', default='', type=str, help='network checkpoint path')
    parser.add_argument('--opt_checkpath', default='', type=str, help='optimizer checkpath')
    parser.add_argument('--net_checkpath2', default='', type=str, help='another network checkpoint path')
        
    # network hyper args
    parser.add_argument('--trainer_mode', default='train', type=str, help='train or test')
    parser.add_argument('--ablation_mode', default='sparse', type=str, help='default sparse, cycle: cycle_sparse')
    parser.add_argument('--loss', default='l1', type=str, help='loss type')
    parser.add_argument('--loss2', default='l2', type=str, help='another loss type')
    parser.add_argument('--network', default='', type=str, help='networkname')
    
    # tester args
    parser.add_argument('--tester_save_name', default='default_save', type=str, help='name of test' )
    parser.add_argument('--tester_save_image', default=False, action='store_true', help='whether to save visualization result' )
    parser.add_argument('--tester_save_path', default='', type=str, help='path for saving tester result' )
    # sparse ct args
    parser.add_argument('--num_views', default=18, type=int, help='common setting: 18/36/72/144 out of 720')
    parser.add_argument('--num_full_views', default=720, type=int, help='720 for fanbeam 2D')
    
    # network args
    parser.add_argument('--net_dict', default='{}', type=str, help='string of dict containing network arguments')
    # freeseed args
    parser.add_argument('--use_mask', default=False, type=bool,)
    parser.add_argument('--soft_mask', default=True, type=bool,)
    return parser


def sparse_main(opt):
    net_name = opt.network
    net2 = None
    print('Network name: ', net_name)
    wrapper_kwargs = {
        'num_views': opt.num_views,
        'num_full_views': opt.num_full_views,
        'img_size': opt.dataset_shape}
    
    if net_name == 'fbp':
        net = nn.Identity()  # only for test
    elif net_name == 'freenet':
        mask_type = 'bp-gaussian-mc'
        net_dict = dict(
            ratio_ginout=0.5, 
            mask_type_init=mask_type, mask_type_down=mask_type,
            fft_size=(opt.dataset_shape, opt.dataset_shape // 2 + 1))
        net_dict.update(eval(opt.net_dict))
        net = FreeNet(1, 1, **net_dict, **wrapper_kwargs)
    elif net_name == 'seednet':
        net_dict = dict(
            ngf=64, n_downsample=1, n_blocks=3, ratio_gin=0.5, ratio_gout=0.5,
            enable_lfu=False, gated=False, global_skip=True)
        net_dict.update(eval(opt.net_dict))
        net = SeedNet(1, 1, **net_dict, **wrapper_kwargs)
    elif net_name == 'dudofreenet':
        mask_type = 'bp-gaussian-mc'
        net_dict = dict(ratio_ginout=0.5, mask_type=mask_type)
        net_dict.update(eval(opt.net_dict))
        print(net_dict)
        net = DuDoFreeNet(**net_dict, **wrapper_kwargs)
    elif 'freeseed' in net_name:
        # use net_dict to specify some arguments for FreeNet
        # use 'freeseed_0.5_1_5' as an example to specify arguments for SeedNet
        mask_type = 'bp-gaussian-mc'
        freenet_dict = dict(
            ratio_ginout=0.5,
            mask_type_init=mask_type, mask_type_down=mask_type,
            fft_size=(opt.dataset_shape, opt.dataset_shape // 2 + 1))
        freenet_dict.update(eval(opt.net_dict))
        
        elems = net_name.split('_')
        ratio = float(elems[1])
        n_downsampling = int(elems[2])
        n_blocks = int(elems[3])
        net = FreeNet(1, 1, **freenet_dict, **wrapper_kwargs)
        net2 = SeedNet(
            1, 1, ratio_gin=ratio, ratio_gout=ratio, n_downsampling=n_downsampling,
            n_blocks=n_blocks, enable_lfu=False, gated=False, **wrapper_kwargs)
    else:
        raise ValueError(f'opt.network selected error, network: {opt.network} not defined')
    
    if opt.trainer_mode == 'train':
        if 'freeseed' in net_name:
            trainer = FreeSeedTrainer(opt=opt, net1=net, net2=net2, loss_type=opt.loss)
        elif net_name in ['freenet', 'seednet']:
            trainer = SimpleTrainer(opt=opt, net=net, loss_type=opt.loss)
        elif net_name in ['dudofreenet']:
            trainer = DuDoFreeNetTrainer(opt=opt, net=net, loss_type=opt.loss)
        trainer.fit()
    elif opt.trainer_mode == 'test':
        tester = SimpleTester(opt=opt, net=net, test_window=None, net2=net2)
        tester.run()
    else:
        raise ValueError('opt trainer mode error: must be train or test, not {}'.format(opt.trainer_mode))

    print('finish')


if __name__ == '__main__':
    parser = get_parser()
    opt = parser.parse_args()
    sparse_main(opt)


