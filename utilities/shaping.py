# This script is mostly stolen from: 
#   https://github.com/RockeyCoss/StructToken/blob/2611e2d438c4edc742475c78ab5095f4bd536515/struct_token/models/utils/shape_convert.py#L1,
# which aims to solve the annoying problem caused by Conv-LayerNorm interaction.
# Modified by Ma Chenglong to adapt to 3D input.
import torch

def rescale_tensor(x, y_min=0., y_max=1., x_min=None, x_max=None):
    x_min = x.min() if x_min is None else x_min
    x_max = x.max() if x_max is None else x_max
    slope = (y_max - y_min) / (x_max - x_min)
    intercept = y_max - slope * x_max
    y = slope * x + intercept
    return y


def threshold_tensor(x, th=None):
    y = torch.zeros_like(x)
    th = x.mean() if th is None else th
    y[x >= th] = 1.0
    return y


def nchw2nlc2nchw(module, x):
    """Flatten [N, C, (D,) H, W] shape tensor `x` to [N, L, C] shape tensor. Use the
    reshaped tensor as the input of `module`, and the convert the output of
    `module`, whose shape is.
    [N, L, C], to [N, C, (D,) H, W].
    Args:
        module: (Callable): A callable object the takes a tensor
            with shape [N, L, C] as input.
        x: (Tensor): The input tensor of shape [N, C, H, W].
    Returns:
        Tensor: The output tensor of shape [N, C, H, W].
    Example:
        >>> import torch
        >>> import torch.nn as nn
        >>> norm = nn.LayerNorm(4)
        >>> feature_map = torch.rand(4, 4, 5, 5)
        >>> output = nchw2nlc2nchw(norm, feature_map)
    """
    x_shape = x.shape
    x = x.flatten(2).transpose(1, 2)  # B, L, C
    x = module(x)
    x = x.transpose(1, 2).reshape(x_shape).contiguous()
    return x


def nlc2nchw2nlc(module, x, shape):
    """Convert [N, L, C] shape tensor `x` to [N, C, H, W] shape tensor. Use the
    reshaped tensor as the input of `module`, and convert the output of
    `module`, whose shape is.
    [N, C, H, W], to [N, L, C].
    Args:
        module: (Callable): A callable object the takes a tensor
            with shape [N, C, H, W] as input.
        x: (Tensor): The input tensor of shape [N, L, C].
        shape: (Sequence[int]): The (depth,) height and width of the
            feature map with shape [N, C, H, W].
    Returns:
        Tensor: The output tensor of shape [N, L, C].
    Example:
        >>> import torch
        >>> import torch.nn as nn
        >>> conv = nn.Conv2d(16, 16, 3, 1, 1)
        >>> feature_map = torch.rand(4, 25, 16)
        >>> output = nlc2nchw2nlc(conv, feature_map, (5, 5))
    """
    def get_prod(elements):
        prod = 1.0
        for elem in elements:
            prod *= elem
        return prod

    assert len(x.shape) == 3
    B, L, C = x.shape
    assert L == get_prod(shape), 'The seq_len doesn\'t match (D,) H, W'
    x = x.transpose(1, 2).reshape(B, C, *shape)  # B, C, D, H, W
    x = module(x)
    x = x.flatten(2).transpose(1, 2)
    return x