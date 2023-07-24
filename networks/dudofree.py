import sys
sys.path.append('..')
from networks.freeseed import FreeNet, SeedNet
from wrappers.basic_wrapper_v2 import BasicSparseWrapper
import numpy as np

import torch
import torch.fft 
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, pool=True):
        super().__init__()
        if pool:
            self.down_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels, out_channels))
        else:
            self.down_conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
                DoubleConv(in_channels, out_channels))
    def forward(self, x):
        return self.down_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class MaskUNet(nn.Module):
    def __init__(self, n_channels, n_classes, pool=True, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64+1, 128, pool=pool)
        self.down2 = Down(128+1, 256, pool=pool)
        self.down3 = Down(256+1, 512, pool=pool)
        factor = 2 if bilinear else 1
        self.down4 = Down(512+1, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        self.pool1 = nn.AvgPool2d(2,2)
        self.pool2 = nn.AvgPool2d(2,2)
        self.pool3 = nn.AvgPool2d(2,2)

    def forward(self, x, mask):
        mask2 = self.pool1(mask)
        mask3 = self.pool2(mask2)
        mask4 = self.pool3(mask3)

        x1 = self.inc(x)
        x2 = self.down1(torch.cat((x1, mask), dim = 1))
        x3 = self.down2(torch.cat((x2, mask2), dim = 1))
        x4 = self.down3(torch.cat((x3, mask3), dim = 1))
        x5 = self.down4(torch.cat((x4, mask4), dim = 1))
        logits = self.up1(x5, x4)
        logits = self.up2(logits, x3)
        logits = self.up3(logits, x2)
        logits = self.up4(logits, x1)
        logits = self.outc(logits)
        return logits


class DuDoFreeNet(BasicSparseWrapper):
    def __init__(
        self, ratio_ginout=0.5, mask_type=None, min_channels=64, 
        num_views=None, img_size=None, num_full_views=720):
        super().__init__(num_views=num_views, img_size=img_size, num_full_views=num_full_views)
        wrapper_kwargs = dict(num_views=num_views, img_size=img_size, num_full_views=num_full_views)
        fft_size = (img_size, img_size//2 + 1)
        freenet_kwargs = dict(
            min_channels=min_channels, ratio_ginout=ratio_ginout, fft_size=fft_size,
            mask_type_init=mask_type, mask_type_down=mask_type)
        self.sino_net = MaskUNet(1, 1)
        self.img_net = FreeNet(2, 1, **freenet_kwargs, **wrapper_kwargs)
    
    def forward(self, sparse_sino, sparse_sino_mask, sparse_mu):
        sino = self.sino_net(sparse_sino, sparse_sino_mask)
        ril_mu = self.radon(sino)
        neg_art, _ = self.img_net(torch.cat((ril_mu, sparse_mu), dim=1))
        pred_mu = neg_art + sparse_mu
        return pred_mu, sino, ril_mu


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    img_size = 256
    num_full_views = 720
    net = DuDoFreeNet(72, img_size=img_size, num_full_views=num_full_views).cuda()
    sino = torch.randn((1, 1, num_full_views, net.det_count)).cuda()
    sino_mask = torch.ones_like(sino).cuda()
    mu = torch.randn((1, 1, img_size, img_size)).cuda()
    
    y = net(sino, sino_mask, mu)
    print([_.shape for _ in y])
    