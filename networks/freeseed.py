import sys
sys.path.append('..')
from modules.basic_layers import (
    get_conv_layer, get_norm_layer, get_act_layer, get_conv_block,
    forward_channel_last
)
from wrappers.basic_wrapper_v2 import BasicSparseWrapper
import torch
import torch.fft 
import torch.nn as nn
import torch.nn.functional as F


def _check_info(x):
    def type2str(t):
        return str(t).replace("'",'').replace('>','').split(' ')[-1]
    if hasattr(x, 'shape'):
        return str(tuple(x.shape))
    else:
        if isinstance(x, (tuple, list)):
            num = str(len(x))
            type1, type2 = type2str(type(x[0])), type2str(type(x[1]))
            return f'len={num}, type=({type1},{type2}), info=({_check_info(x[0])}, {_check_info(x[1])})'
        else:
            return type2str(type(x))


def gen_gaussian_bandpass(center=0, width=0.2, shape=(256, 129), 
    shift_to_fit_fft=True, unsqueeze=True, threshold=False):
    # accept input format: [C, 1]
    def _make_shape(x):
        if not torch.is_tensor(x):
            if not isinstance(x, (tuple, list)):
                x = torch.tensor([x])
            else:
                x = torch.tensor(x)
        if x.ndim <= 3:
            dim = x.numel()
            x = x.reshape(dim, 1, 1)
        return x

    center = _make_shape(center).clamp(0, 1)
    width = _make_shape(width).clamp(min=1e-12, max=2)
    assert center.shape == width.shape, f'center: {center} != width: {width}.'
    X, Y = torch.meshgrid(torch.arange(shape[0]), torch.arange(shape[1]))
    X = torch.repeat_interleave(X.unsqueeze(0), center.shape[0], dim=0).to(center.device)
    Y = torch.repeat_interleave(Y.unsqueeze(0), center.shape[0], dim=0).to(center.device)
    x0 = (shape[0]-1) // 2
    y0 = 0
    D2 = ((X - x0) ** 2 + (Y - y0) ** 2).float()
    D2 /= D2.max()
    H = torch.exp(-((D2 - center ** 2)/(D2.sqrt() * width + 1e-12)) ** 2)
    H = torch.roll(H, H.shape[-2]//2 + 1, -2) if shift_to_fit_fft else H
    H = H.unsqueeze(0) if unsqueeze else H
    if threshold:
        H_mean = H.mean()
        H[H < H_mean] = 0.0
        H[H >= H_mean] = 1.0
    return H


class FourierUnit(nn.Module):
    def __init__(
        self, in_channels, out_channels, groups=1, spatial_scale_factor=None, spatial_scale_mode='cubic',
        spectral_pos_encoding=False, fft_norm='ortho', norm_type='BATCH', act_type='RELU', act_kwargs=None, 
        mask_type=None, fft_size=(256, 129)):
        # bn_layer not used
        super(). __init__()
        self.groups = groups
        ffc_in_channels = in_channels * 2 + (2 if spectral_pos_encoding else 0)
        self.conv = get_conv_layer(
            in_channels=ffc_in_channels,
            out_channels=out_channels * 2,
            kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        act_kwargs = {} if act_kwargs is None else act_kwargs
        self.norm = get_norm_layer(norm_type=norm_type, args=[out_channels * 2])
        self.act = get_act_layer(act_type=act_type, kwargs=act_kwargs)

        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.fft_norm = fft_norm
        if mask_type is not None:
            if mask_type == 'global':
                #self.mask = nn.Parameter(torch.randn((1, 1, *fft_size), dtype=torch.float32) * 0.02, requires_grad=True)
                self.mask = nn.Parameter(torch.randn((1, ffc_in_channels, *fft_size), dtype=torch.float32) * 0.02, requires_grad=True)
                print('...Using global mask...')
            elif 'bp-gaussian-sc' in mask_type:
                self.mask = None
                self.center = nn.Parameter(torch.tensor([0.], dtype=torch.float32), requires_grad=True)
                self.width = nn.Parameter(torch.tensor([1.], dtype=torch.float32), requires_grad=True)
                print('...Using single-channel Gaussian bandpass...')
            elif 'bp-gaussian-mc' in mask_type:
                self.mask = None
                self.center = nn.Parameter(torch.tensor([0. for _ in range(ffc_in_channels)], dtype=torch.float32), requires_grad=True)
                self.width = nn.Parameter(torch.tensor([1. for _ in range(ffc_in_channels)], dtype=torch.float32), requires_grad=True)
                print('Initialized: ', (self.center.mean().item(), self.width.mean().item()))
                print('...Using multi-channel Gaussian bandpass...')
            else:
                self.center, self.width, self.mask = None, None, None
        else:
            self.center, self.width, self.mask = None, None, None
        
        self.mask_type = mask_type


    def forward(self, x):
        batch = x.shape[0]
        if self.spatial_scale_factor is not None:
            orig_size = x.shape[2:]
            x = F.interpolate(x, scale_factor=self.spatial_scale_factor, mode=self.spatial_scale_mode, align_corners=False)

        # FFC convolution
        fft_dim = (-2, -1)  # (batch, c, h, w/2+1, 2)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)  # (batch, c, h, w/2+1, 2)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])  # (batch, 2c, h, w/2+1)

        if self.mask is not None or self.center is not None or self.width is not None:
            if 'gaussian' in self.mask_type:
                mask = gen_gaussian_bandpass(self.center, self.width, shape=ffted.shape[2:])
            else:
                mask = self.mask 
                
            if mask.shape[-2] != ffted.shape[-2] or mask.shape[-1] != ffted.shape[-1]:
                mask = F.interpolate(mask, size=ffted.shape[2:], mode='bilinear', align_corners=False)
            ffted = ffted * mask

        if self.spectral_pos_encoding:
            height, width = ffted.shape[-2:]
            coords_vertical = torch.linspace(0, 1, height)[None, None, :, None].expand(batch, 1, height, width).to(ffted)
            coords_horizontal = torch.linspace(0, 1, width)[None, None, None, :].expand(batch, 1, height, width).to(ffted)
            ffted = torch.cat((coords_vertical, coords_horizontal, ffted), dim=1)

        ffted = self.conv(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.act(self.norm(ffted))
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[2:]
        out = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)
        if self.spatial_scale_factor is not None:
            out = F.interpolate(out, size=orig_size, mode=self.spatial_scale_mode, align_corners=False)
        return out


class SpectralTransform(nn.Module):
    """
      ___________________________add____________________________
     |                                                          |
     |                                                          v
    conv-norm-act -> real FFT2D -> conv-norm-act -> real iFFT2D -> conv1x1
    """
    def __init__(
        self, in_channels, out_channels, stride=1, groups=1, 
        norm_type='BATCH', act_type='LRELU', act_kwargs=None,
        enable_lfu=False, **fu_kwargs):
        super().__init__()
        self.enable_lfu = enable_lfu
        self.downsample = nn.AvgPool2d(kernel_size=stride, stride=stride) if stride == 2 else nn.Identity()
        self.stride = stride
        act_kwargs = {} if act_kwargs is None else act_kwargs
        self.conv1 = get_conv_block(
            in_channels, out_channels//2, kernel_size=1, groups=groups, bias=False, 
            norm_type=norm_type, act_type=act_type, act_kwargs=act_kwargs, adn_order='CNA',)

        self.fu = FourierUnit(out_channels // 2, out_channels // 2, groups=groups, norm_type=norm_type, 
            act_type=act_type, act_kwargs=act_kwargs, **fu_kwargs)
        if self.enable_lfu:
            self.lfu = FourierUnit(out_channels // 2, out_channels // 2, groups=groups)
        self.conv2 = get_conv_layer(out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv1(x)
        out = self.fu(x)

        if self.enable_lfu:
            C, H = x.shape[1:3]
            num_split = 2
            split_size = H // num_split
            xs = torch.cat(torch.split(x[:, :C // 4], split_size, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_size, dim=-1), dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, num_split, num_split).contiguous()
        else:
            xs = 0
        out = self.conv2(x + out + xs)
        return out


class FFC(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, ratio_gin, ratio_gout, stride=1, padding=0,
        dilation=1, groups=1, bias=False, is_transposed=False, padding_mode='reflect', 
        enable_lfu=False, gated=False, mask_type=None, fft_size=(256, 129)):
        super().__init__()
        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride
        
        in_channels_global = int(in_channels * ratio_gin)
        in_channels_local = in_channels - in_channels_global
        out_channels_global = int(out_channels * ratio_gout)
        out_channels_local = out_channels - out_channels_global
        self.gated = gated
        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_channels_global
        conv_op = nn.ConvTranspose2d if is_transposed else nn.Conv2d
        
        module = nn.Identity if in_channels_local == 0 or out_channels_local == 0 else conv_op
        self.convl2l = module(in_channels_local, out_channels_local, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_mode)
        module = nn.Identity if in_channels_local == 0 or out_channels_global == 0 else conv_op
        self.convl2g = module(in_channels_local, out_channels_global, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_mode)
        module = nn.Identity if in_channels_global == 0 or out_channels_local == 0 else conv_op
        self.convg2l = module(in_channels_global, out_channels_local, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_mode)
        
        # TODO: adapt transposed conv to this type:
        module = nn.Identity if in_channels_global == 0 or out_channels_global == 0 else SpectralTransform
        self.convg2g = module(
            in_channels_global, out_channels_global, stride=stride, groups=groups if groups == 1 else groups // 2, 
            enable_lfu=enable_lfu, mask_type=mask_type, fft_size=fft_size)
        module = nn.Identity if in_channels_global == 0 or out_channels_local == 0 or not gated else nn.Conv2d
        self.gate = module(in_channels, 2, 1)  # 1*1 conv

    def forward(self, x):
        x_local, x_global = x if isinstance(x, tuple) else (x, 0)
        out_local, out_global = 0, 0
        if self.gated:
            total_input_parts = [x_local]
            if torch.is_tensor(x_global):
                total_input_parts.append(x_global)
            total_input = torch.cat(total_input_parts, dim=1)

            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate, l2g_gate = 1, 1

        if self.ratio_gout != 1:
            out_local = self.convl2l(x_local) + self.convg2l(x_global) * g2l_gate
        if self.ratio_gout != 0:
            out_global = self.convl2g(x_local) * l2g_gate + self.convg2g(x_global)

        return out_local, out_global


class FFCBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, 
        adn_order='NA', act_type='LRELU', act_kwargs=None, norm_type='BATCH', padding_mode='reflect',
        ratio_gin=0.5, ratio_gout=0.5, enable_lfu=False, gated=False, mask_type=None, fft_size=(256, 129)):
        super().__init__()
        assert isinstance(adn_order, str)
        adn_order = adn_order.upper()
        adn_order = 'C' + adn_order if 'C' not in adn_order else adn_order  # default first layer as convolution
        out_channels_global = int(out_channels * ratio_gout)
        self.adn_order = adn_order
        self.ffc = FFC(
            in_channels, out_channels, kernel_size, ratio_gin=ratio_gin, ratio_gout=ratio_gout, 
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False, enable_lfu=enable_lfu,
            padding_mode=padding_mode, gated=gated, mask_type=mask_type, fft_size=fft_size)
        
        self.norm_local = nn.Identity() if ratio_gout == 1 else get_norm_layer(norm_type, [out_channels - out_channels_global])
        self.norm_global = nn.Identity() if ratio_gout == 0 else get_norm_layer(norm_type, [out_channels_global])
        self.act_local = nn.Identity() if ratio_gout == 1 else get_act_layer(act_type, kwargs=act_kwargs)
        self.act_global = nn.Identity() if ratio_gout == 0 else get_act_layer(act_type, kwargs=act_kwargs)
    
    @staticmethod
    def forward_norm(layer, x):
        return forward_channel_last(layer, x) if isinstance(layer, nn.LayerNorm) else layer(x)

    def forward(self, x):
        x_local, x_global = self.ffc(x)
        x_local = self.forward_norm(self.norm_local, x_local)
        x_global = self.forward_norm(self.norm_global, x_global)
        x_local = self.act_local(x_local)
        x_global = self.act_global(x_global)
        return x_local, x_global
    

class MultipleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(
        self, in_channels, out_channels, mid_channels=None, kernel_size=3, stride=1, padding=1, use_ffc=False,
        norm_type='BATCH', act_type='LRELU', act_kwargs=None, num_convs=2, **fu_kwargs):
        super().__init__()
        fu_kwargs_mask = {'ratio_gin':0.5, 'ratio_gout':0.5, 'enable_lfu':False, 
            'mask_type':None, 'fft_size':(256,129)}
        fu_kwargs_mask.update(fu_kwargs)
        self.use_ffc = use_ffc
        self.ratio_gin = fu_kwargs_mask['ratio_gin']
        self.ratio_gout = fu_kwargs_mask['ratio_gout']
        mid_channels = out_channels if mid_channels is None else mid_channels
        
        conv_list = []
        for i in range(num_convs):
            cur_in_channels = in_channels if i == 0 else mid_channels
            cur_out_channels = out_channels if i == (num_convs - 1) else mid_channels
            if use_ffc:
                block = FFCBlock(cur_in_channels, cur_out_channels, kernel_size, stride=stride, padding=padding, 
                act_type=act_type, act_kwargs=act_kwargs, **fu_kwargs_mask)
            else:
                block = get_conv_block(cur_in_channels, cur_out_channels, kernel_size=kernel_size, stride=stride,
                padding=padding, norm_type=norm_type, act_type=act_type, act_kwargs=act_kwargs)
            conv_list.append(block)
        self.conv_blocks = nn.Sequential(*conv_list) if num_convs > 1 else conv_list[0]

    def forward(self, x):
        y = self.conv_blocks(x)
        return y  # return a tuple if use_ffc is True


class EncodingBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, pooling=True, norm_type='BATCH', act_type='LRELU', act_kwargs=None,
        use_ffc=False, ratio_gin=0, ratio_gout=0, enable_lfu=False, mask_type=None, fft_size=(256,129)):
        super().__init__()
        self.pooling = pooling
        conv_kwargs = {'kernel_size': kernel_size, 'norm_type': norm_type, 'act_type': act_type, 'act_kwargs': act_kwargs,}
        fu_kwargs = {'ratio_gin': ratio_gin, 'ratio_gout': ratio_gout, 'enable_lfu': enable_lfu, 
            'mask_type':mask_type, 'fft_size':fft_size}
        if pooling:
            self.down = nn.MaxPool2d(2)
            self.conv = MultipleConv(in_channels, out_channels, use_ffc=use_ffc, num_convs=2, **conv_kwargs, **fu_kwargs)
        else:
            self.down = MultipleConv(in_channels, in_channels, kernel_size=3, stride=2, padding=1, norm_type=norm_type, 
                act_type=act_type, act_kwargs=act_kwargs, use_ffc=use_ffc, num_convs=1, **fu_kwargs)
            self.conv = MultipleConv(in_channels, out_channels, use_ffc=use_ffc, num_convs=1, **conv_kwargs, **fu_kwargs)

    def forward(self, x):
        if isinstance(x, tuple):
            x_local, x_global = x
            if self.pooling:
                x_local, x_global = self.down(x_local), self.down(x_global)
            else:
                x_local, x_global = self.down(x)
            out = self.conv((x_local, x_global))
        else:
            out = self.conv(self.down(x))
        return out
    

class DecodingBlock(nn.Module):
    # Fourier encoding, non-Fourier decoding
    def __init__(self, in_channels, out_channels, kernel_size=3, bilinear=True, norm_type='BATCH', act_type='RELU', act_kwargs=None,
        skip_mode=None):
        super().__init__()
        conv_kwargs = {'kernel_size': kernel_size, 'norm_type': norm_type, 'act_type': act_type, 'act_kwargs': act_kwargs}
        fu_kwargs = {'ratio_gin': 0, 'ratio_gout': 0, 'enable_lfu': False, 
            'mask_type':None, 'fft_size':(256,129), 'mask_sigmas':(1.0, 0.0)}
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = MultipleConv(in_channels, in_channels//2, use_ffc=False, num_convs=1, **conv_kwargs, **fu_kwargs)
            self.merge = MultipleConv(in_channels, out_channels, in_channels//2, use_ffc=False, **conv_kwargs, **fu_kwargs)
        else:  # TODO: adapt transposed conv to Fourier conv
            self.up = nn.Identity()
            self.conv = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
            self.merge = MultipleConv(in_channels, out_channels, use_ffc=False, **conv_kwargs, **fu_kwargs)
        
        self.skip_mode = skip_mode
        if skip_mode == 'conv':
            self.skip_block = MultipleConv(in_channels//2, in_channels//2, use_ffc=False, num_convs=1, **conv_kwargs, **fu_kwargs)
        elif skip_mode == 'fourier':
            self.skip_block = MultipleConv(in_channels//2, in_channels//2, use_ffc=True, num_convs=1, **conv_kwargs, **fu_kwargs)
        else:
            self.skip_block = None
    
    def pad_to_higher_resolution(self, x_low, x_high):
        diffY = x_high.shape[2] - x_low.shape[2]
        diffX = x_high.shape[3] - x_low.shape[3]
        return F.pad(x_low, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

    def forward(self, x1, x2):  # x1 is the feature to be upsampled, x2 is the skip
        if isinstance(x1, tuple):  # bottleneck
            x1 = torch.cat(x1, dim=1)

        if self.skip_mode == 'conv':
            x2 = torch.cat(x2, dim=1) if isinstance(x2, tuple) else x2
            x2 = self.skip_block(x2)
        elif self.skip_mode == 'fourier':
            x2 = self.skip_block(x2) if isinstance(x2, tuple) else x2
            x2 = torch.cat(x2, dim=1)
        else:
            x2 = torch.cat(x2, dim=1) if isinstance(x2, tuple) else x2
        
        x1 = self.up(x1)
        x1 = self.conv(x1)
        x1 = self.pad_to_higher_resolution(x1, x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.merge(x)
        return x


class FFCResNetBlock(nn.Module):
    def __init__(
        self, dim, padding_mode, norm_type='BATCH', act_type='RELU', kernel_size=3, padding=1, 
        dilation=1, inline=False, hidden_dim=None, out_dim=None, **conv_kwargs):
        super().__init__()
        hidden_dim = dim if hidden_dim is None else hidden_dim
        out_dim = dim if out_dim is None else out_dim
        self.conv1 = FFCBlock(
            dim, hidden_dim, kernel_size=kernel_size, padding=padding, dilation=dilation, 
            norm_type=norm_type, act_type=act_type, padding_mode=padding_mode, **conv_kwargs)
        self.conv2 = FFCBlock(
            hidden_dim, out_dim, kernel_size=kernel_size, padding=padding, dilation=dilation, 
            norm_type=norm_type, act_type=act_type, padding_mode=padding_mode, **conv_kwargs)
        self.inline = inline
    
    def forward(self, x):
        if self.inline:
            x_l, x_g = x[:, :-self.conv1.ffc.global_in_num], x[:, -self.conv1.ffc.global_in_num:]
        else:
            x_l, x_g = x if isinstance(x, tuple) else (x, 0)

        id_l, id_g = x_l, x_g
        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))
        x_l = id_l + x_l
        x_g = id_g + x_g
        
        out = torch.cat((x_l, x_g), dim=1) if self.inline else (x_l, x_g)
        return out
    

class FreeNet(BasicSparseWrapper):
    def __init__(
        self, in_channels, out_channels, factor=2, min_channels=32, max_channels=1024, num_in_conv=2, num_stages=5,
        norm_type='BATCH', act_type='RELU', act_kwargs=None, bilinear=True, skip_mode=None, 
        use_fft=True, ratio_ginout=0.5, mask_type_init=None, mask_type_down=None, fft_size=(256, 129),
        **wrapper_kwargs):
        super().__init__(**wrapper_kwargs)
        feature_channels = [min_channels * (factor ** i) for i in range(num_stages)]  # [32, 64, 128, 256, 512]
        feature_channels = [min(max(c, min_channels), max_channels) for c in feature_channels]
        feature_channels_rev = feature_channels[::-1]
        
        self.out_channels = feature_channels[-1]
        self.num_stages = num_stages
        down_in_channels = feature_channels[:-1]  # [32, 64, 128, 256]
        down_out_channels = feature_channels[1:]  # [64, 128, 256, 512]
        up_in_channels = feature_channels_rev[:-1]  # [512, 256, 128, 64]
        up_out_channels = feature_channels_rev[1:]  # [256, 128, 64, 32]

        conv_kwargs = {'kernel_size': 3, 'norm_type': norm_type, 'act_type': act_type, 'act_kwargs': act_kwargs}
        init_fu_kwargs = {'ratio_gin': 0, 'ratio_gout': ratio_ginout, 'enable_lfu': False, 
            'mask_type':mask_type_init, 'fft_size':fft_size}
        down_fu_kwargs = {'ratio_gin': ratio_ginout, 'ratio_gout': ratio_ginout, 'enable_lfu': False,
            'mask_type':mask_type_down, 'fft_size':fft_size}

        self.inc = nn.Sequential(
            MultipleConv(in_channels, feature_channels[0], num_convs=1, use_ffc=use_fft, **conv_kwargs, **init_fu_kwargs),
            MultipleConv(feature_channels[0], feature_channels[0], num_convs=num_in_conv-1, use_ffc=use_fft, **conv_kwargs, **down_fu_kwargs),
        )
        self.down_blocks = nn.ModuleList(
            [EncodingBlock(down_in_channels[i], down_out_channels[i], use_ffc=use_fft, **conv_kwargs, **down_fu_kwargs) 
             for i in range(num_stages-1)])
        self.up_blocks = nn.ModuleList(
            [DecodingBlock(up_in_channels[i], up_out_channels[i], bilinear=bilinear, skip_mode=skip_mode, 
            **conv_kwargs) for i in range(num_stages-1)])
        self.outc = nn.Conv2d(feature_channels[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        features = [0] * self.num_stages
        y = self.inc(x)
        features[0] = y
        for i in range(self.num_stages-1):
            y = self.down_blocks[i](y)
            features[i+1] = y
        
        for i in range(self.num_stages-1):
            skip = features[self.num_stages-i-2]
            y = self.up_blocks[i](y, skip)
        
        y = torch.cat(y, dim=1) if isinstance(y, tuple) else y
        y = self.outc(y)  # negative artifact
        return y, (y + x[:, 0:1])
    
    
class SeedNet(BasicSparseWrapper):
    def __init__(
        self, in_channels, out_channels, ngf=64, n_downsampling=1, n_blocks=3, norm_type='BATCH', 
        padding_mode='reflect', act_type='RELU', max_features=1024, out_ffc=False, global_skip=True, 
        ratio_gin=0.5, ratio_gout=0.5, enable_lfu=False, gated=False, latent_conv_kwargs=None, **wrapper_kwargs):
        super().__init__(**wrapper_kwargs)
        assert (n_blocks >= 0)
        self.global_skip = global_skip
        init_conv_kwargs = {'ratio_gin':0,'ratio_gout':0,'enable_lfu':False}
        downsample_conv_kwargs = {'ratio_gin':0, 'ratio_gout':0,'enable_lfu':False} 
        latent_conv_kwargs = {'ratio_gin':ratio_gin, 'ratio_gout':ratio_gout, 'enable_lfu':enable_lfu, 'gated':gated} if latent_conv_kwargs is None else latent_conv_kwargs
        
        self.inc = nn.Sequential(
            nn.ReflectionPad2d(3),
            FFCBlock(in_channels, ngf, kernel_size=7, padding=0, norm_type=norm_type, act_type=act_type, **init_conv_kwargs)
        )
        
        # downsampling
        self.downsample = []
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                cur_conv_kwargs = dict(downsample_conv_kwargs)
                cur_conv_kwargs['ratio_gout'] = latent_conv_kwargs.get('ratio_gin', 0)
            else:
                cur_conv_kwargs = downsample_conv_kwargs
            
            print(f'downsampling {i}:', cur_conv_kwargs)
            cur_in_ch = min(max_features, ngf * mult)
            cur_out_ch = min(max_features, ngf * mult * 2)
            self.downsample.append(FFCBlock(
                cur_in_ch, cur_out_ch, kernel_size=3, stride=2, padding=1,
                norm_type=norm_type, act_type=act_type, **cur_conv_kwargs))
        self.downsample = nn.Sequential(*self.downsample)
        
        mult = 2 ** n_downsampling
        num_feats_bottleneck = min(max_features, ngf * mult)

        # latent block
        self.blocks = []
        for i in range(n_blocks):
            cur_resblock = FFCResNetBlock(
                num_feats_bottleneck, padding_mode=padding_mode, act_type=act_type,
                norm_type=norm_type, **latent_conv_kwargs)
            self.blocks.append(cur_resblock)
        self.blocks = nn.ModuleList(self.blocks)
        
        # upsample
        self.upsample = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            self.upsample += [nn.ConvTranspose2d(
                min(max_features, ngf * mult),
                min(max_features, int(ngf * mult / 2)),
                kernel_size=3, stride=2, padding=1, output_padding=1),
                get_norm_layer(norm_type=norm_type, args=min(max_features, int(ngf * mult / 2))),
                get_act_layer(act_type=act_type)]
        self.upsample = nn.Sequential(*self.upsample)

        self.outc = []
        if out_ffc:
            self.outc.append(FFCResNetBlock(ngf, padding_mode=padding_mode, norm_type=norm_type, act_type=act_type, inline=True))
        self.outc += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, kernel_size=7, padding=0)]
        self.outc = nn.Sequential(*self.outc)
    
    def forward(self, x):
        #print('Input:',_check_info(x))
        y0 = self.inc(x)
        #print('Stem:',_check_info(y0))
        y = self.downsample(y0)
        #print('Downsample:',_check_info(y))
        for i in range(len(self.blocks)):
            y = self.blocks[i](y)
            #print(f'Block {i}:',_check_info(y))
        if isinstance(y, tuple):
            assert torch.is_tensor(y[0]) or torch.is_tensor(y[1])
            y = torch.cat(y, dim=1) if torch.is_tensor(y[1]) else y[0]
        #print('After blocks:',_check_info(y))
        y = self.upsample(y)
        #print('Upsample:',_check_info(y))
        y = self.outc(y)
        #print('Output:',_check_info(y))
        return (y + x) if self.global_skip else y



if __name__ == '__main__':
    img_size = 256
    fft_size = (img_size, img_size//2 + 1)
    img = torch.randn((1, 1, img_size, img_size))
    
    net = FreeNet(
        1, 1, ratio_ginout=0.5, min_channels=64, max_channels=1024,
        norm_type='BATCH', act_type='RELU', global_skip=True, 
        num_stages=5, skip_mode=None, fft_size=fft_size,
        num_views=72, img_size=img_size).cuda()  # 31M/63G
    net2 = SeedNet(1, 1, ratio_gin=0.5, ratio_gout=0.5).cuda()
    
    print(net(img.cuda()).shape)
    print(net2(img.cuda()).shape)
    