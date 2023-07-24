# coding=utf-8 #
import sys
sys.path.append('..')
from modules.basic_layers import *
import torch.nn.functional as F


class BasicConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, is_transposed=False, output_padding=0, adn_order='NAD', act_type='RELU', act_kwargs=None,
                 norm_type='BATCH', drop_type='DROPOUT', drop_rate=0., dropout_dims=1, spatial_dims=2, **kwargs):
        super().__init__()
        assert isinstance(adn_order, str)
        adn_order = adn_order.upper()
        adn_order = 'C' + adn_order if 'C' not in adn_order else adn_order  # default first layer as convolution
        self.adn_order = adn_order
        self.conv = get_conv_layer(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, 
            groups=groups, bias=bias, is_transposed=is_transposed, output_padding=output_padding, spatial_dims=spatial_dims)
        self.norm = get_norm_layer(norm_type, [out_channels], spatial_dims=spatial_dims) if norm_type is not None else nn.Identity()
        self.act = get_act_layer(act_type, kwargs=act_kwargs) if act_type is not None else nn.Identity()
        self.drop = get_drop_layer(drop_type=drop_type, drop_rate=drop_rate, spatial_dims=dropout_dims)

    @staticmethod
    def forward_norm(layer, x):
        if isinstance(layer, nn.LayerNorm):
            y = forward_channel_last(layer, x)
        else:
            y = layer(x)
        return y

    def forward(self, x):
        y = x
        layer_dict = {'C': self.conv, 'N': self.norm, 'A': self.act, 'D': self.drop}
        for name in self.adn_order:
            layer = layer_dict[name]
            y = self.forward_norm(layer, y)
        return y


class BasicResBlock(nn.Module):
    # Adapted from Huang Ziyan.
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, 
                 norm_type='BATCH', act_type='RELU', act_kwargs=None, is_transposed=False, output_padding=0, 
                 spatial_dims=2, **kwargs):
        super().__init__()
        # fixed activation and normalization
        # act_type = 'LEAKY'
        # norm_type = 'INSTANCE'
        norm_kwargs = {'affine':True}

        self.conv1 = get_conv_layer(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, 
            groups=groups, bias=bias, is_transposed=is_transposed, output_padding=output_padding, spatial_dims=spatial_dims)
        self.norm1 = get_norm_layer(norm_type, args=[out_channels], spatial_dims=spatial_dims, kwargs=norm_kwargs)
        self.act1 = get_act_layer(act_type=act_type, kwargs=act_kwargs)

        self.conv2 = get_conv_layer(
            out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, 
            groups=groups, bias=bias, is_transposed=is_transposed, output_padding=output_padding, spatial_dims=spatial_dims)
        self.norm2 = get_norm_layer(norm_type, args=[out_channels], spatial_dims=spatial_dims, kwargs=act_kwargs)
        self.act2 = get_act_layer(act_type=act_type, kwargs=act_kwargs)
        
        if in_channels != out_channels:
            self.conv3 = get_conv_layer(in_channels, out_channels, kernel_size=1, stride=stride, spatial_dims=spatial_dims)
        else:
            self.conv3 = None
        
    def forward(self, x):
        y = self.conv1(x)
        if isinstance(self.norm1, nn.LayerNorm):
            y = forward_channel_last(self.norm1, y)
        else:
            y = self.norm1(y)
        y = self.act1(y)

        y = self.conv2(y)
        if isinstance(self.norm2, nn.LayerNorm):
            y = forward_channel_last(self.norm2, y)
        else:
            y = self.norm2(y)

        x = self.conv3(x) if self.conv3 is not None else x
        return self.act2(y + x)


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt implementation
    <https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py>
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, bias=True, expand=False,
                 expand_ratio=4, skip_connection=False, act_type='RELU', act_kwargs=None, norm_type='LAYER',
                 drop_type='PATH', drop_rate=0., dropout_dims=1, spatial_dims=2, layer_scale_init=1e-6):
        super().__init__()
        dw_padding = (kernel_size - 1) // 2

        self.dwconv = get_conv_layer(
            in_channels, in_channels, kernel_size=kernel_size, padding=dw_padding, 
            groups=in_channels, dilation=dilation, bias=bias, spatial_dims=spatial_dims)
        self.norm = get_norm_layer(norm_type, [in_channels], spatial_dims=spatial_dims) if norm_type is not None else nn.Identity()

        if expand:
            hidden_channels = int(in_channels * expand_ratio)
            self.pwconv = nn.Sequential(
                get_conv_layer(in_channels, hidden_channels, kernel_size=1, spatial_dims=spatial_dims),
                get_act_layer(act_type, kwargs=act_kwargs) if act_type is not None else nn.Identity(),
                get_conv_layer(hidden_channels, out_channels, kernel_size=1, spatial_dims=spatial_dims))
        else:
            self.pwconv = nn.Sequential(
                get_act_layer(act_type, kwargs=act_kwargs) if act_type is not None else nn.Identity(),
                get_conv_layer(in_channels, out_channels, kernel_size=1, spatial_dims=spatial_dims))

        self.gamma = nn.Parameter(layer_scale_init * torch.ones((1, out_channels, 1, 1)), requires_grad=True) if layer_scale_init > 0 else None
        self.drop = get_drop_layer(drop_type=drop_type, drop_rate=drop_rate, spatial_dims=dropout_dims)
        if skip_connection:
            if in_channels != out_channels:
                self.skip = get_conv_layer(in_channels, out_channels, kernel_size=1, bias=False, spatial_dims=spatial_dims)
            else:
                self.skip = nn.Identity()
        else:
            self.skip = None

    def forward(self, x):
        out = self.dwconv(x)
        out = out.permute(0, 2, 3, 1) if isinstance(self.norm, nn.LayerNorm) else out  # channel last
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2) if isinstance(self.norm, nn.LayerNorm) else out  # channel first
        out = self.pwconv(out)
        out = self.gamma * out if self.gamma is not None else out
        out = self.skip(x) + self.drop(out) if self.skip is not None else self.drop(out)
        return out


class DepthwiseConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, adn_order='NAD', norm_type='INSTANCE', act_type='RELU',
                 act_kwargs=None, drop_rate=0., dropout_dims=1, with_point_wise=True, bias=False, spatial_dims=2):
        super().__init__()
        if isinstance(kernel_size, (tuple, list)):
            padding = [(k-1)//2 for k in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        self.depth_wise = get_conv_block(
            in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels,
            act_type=act_type, act_kwargs=act_kwargs, norm_type=norm_type, drop_rate=drop_rate, dropout_dims=dropout_dims,
            adn_order=adn_order, bias=bias, spatial_dims=spatial_dims)

        self.point_wise = get_conv_block(
            in_channels, out_channels, kernel_size=1, act_type=act_type, act_kwargs=act_kwargs,
            norm_type=norm_type, drop_rate=drop_rate, dropout_dims=dropout_dims, adn_order=adn_order,
            bias=bias, spatial_dims=spatial_dims) if with_point_wise else None

    def forward(self, x):
        out = self.depth_wise(x)
        if self.point_wise is not None:
            out = self.point_wise(out)
        return out


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=1, pool_type='avg', act_type = 'sigmoid'):
        super(SEBlock, self).__init__()

        if in_channels < reduction:
            expand_channels = in_channels * reduction
            self.expand = nn.Conv2d(in_channels, expand_channels, kernel_size=1)
        else:
            expand_channels = in_channels
            self.expand = None

        self.squeeze = nn.AdaptiveAvgPool2d(1) if pool_type == 'avg' else nn.AdaptiveMaxPool2d(1)

        self.excitation = nn.Sequential(
            nn.Linear(expand_channels, expand_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            )

        if act_type == 'sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.expand(x) if self.expand is not None else x

        y = self.squeeze(y)
        y = y.reshape(y.shape[0], y.shape[1])
        y = self.excitation(y).reshape(b, c, 1, 1)
        y = self.act(y)
        return x * y.expand_as(x)


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(CBAM, self).__init__()
        
        self.cam = ChannelAttention(in_channels, reduction=reduction)
        self.sam = SpatialAttention(kernel_size=7)

    def forward(self, x):
        x = self.cam(x) * x
        y = self.sam(x) * x
        return y

# github: https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16, act_type = 'sigmoid'):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False))
        if act_type == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            self.act = nn.Tanh()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.act(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, act_type = 'sigmoid',):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        if act_type == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            self.act = nn.Tanh()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.act(x)



class CoordAttnBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mid_channels = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.ReLU(inplace=True)
        
        self.conv_h = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        H, W = x.shape[2:]
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) * y
        
        x_h, x_w = torch.split(y, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
    
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        return x * a_w * a_h



class DummyNet:
    def __init__(self): pass

    def state_dict(self): return None

    def train(self): return self

    def eval(self): return self

    def step(self): pass

    def to(self): return self

