import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import math

class SEBlock3D(nn.Module):
    """Squeeze-and-Excitation block for 3D"""
    def __init__(self, channels, reduction=4):
        super(SEBlock3D, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.SiLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class MBConv3D(nn.Module):
    """Mobile Inverted Residual Bottleneck for 3D"""
    def __init__(self, in_channels, out_channels, expand_ratio=4, stride=1, se_ratio=0.25):
        super(MBConv3D, self).__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = int(in_channels * expand_ratio)
        
        layers = []
        # Expansion phase
        if expand_ratio != 1:
            layers.extend([
                nn.Conv3d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.SiLU()
            ])
        
        # Depthwise convolution
        layers.extend([
            nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm3d(hidden_dim),
            nn.SiLU()
        ])
        
        # Squeeze-and-Excitation
        if se_ratio > 0:
            layers.append(SEBlock3D(hidden_dim, reduction=int(1/se_ratio)))
        
        # Projection phase
        layers.extend([
            nn.Conv3d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)

class CNN(nn.Module):
    def __init__(self, input_dim=40, input_size=(65, 77, 49), num_classes=2, width_mult=1.0, depth_mult=1.0):
        super(CNN, self).__init__()
        self.input_dim = input_dim
        
        # Stem
        stem_channels = self._scale_channels(32, width_mult)
        self.stem = nn.Sequential(
            nn.Conv3d(input_dim, stem_channels, 3, 2, 1, bias=False),
            nn.BatchNorm3d(stem_channels),
            nn.SiLU()
        )
        
        # Building blocks with compound scaling
        # [expand_ratio, channels, num_blocks, stride]
        block_configs = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 40, 2, 2],
            [6, 80, 2, 1],
        ]
        
        self.blocks = nn.ModuleList()
        in_channels = stem_channels
        
        for expand_ratio, channels, num_blocks, stride in block_configs:
            out_channels = self._scale_channels(channels, width_mult)
            num_blocks = self._scale_depth(num_blocks, depth_mult)
            
            for i in range(num_blocks):
                s = stride if i == 0 else 1
                self.blocks.append(
                    MBConv3D(in_channels, out_channels, expand_ratio, s)
                )
                in_channels = out_channels
        
        # Head
        head_channels = self._scale_channels(320, width_mult)
        self.head = nn.Sequential(
            nn.Conv3d(in_channels, head_channels, 1, bias=False),
            nn.BatchNorm3d(head_channels),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d(1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(head_channels, num_classes)
        )
    
    def _scale_channels(self, channels, width_mult):
        """Scale channel dimensions"""
        return int(math.ceil(channels * width_mult))
    
    def _scale_depth(self, depth, depth_mult):
        """Scale number of blocks"""
        return int(math.ceil(depth * depth_mult))
    
    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

