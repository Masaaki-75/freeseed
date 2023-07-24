import sys
sys.path.append('..')
import numpy as np
import torch.nn as nn
from modules.basic_layers import get_act_layer
from einops import rearrange, repeat
# from monai.networks.blocks import MLPBlock


class MLPBlock(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_type='GELU', drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = get_act_layer(act_type)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
