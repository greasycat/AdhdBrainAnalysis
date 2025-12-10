import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, input_dim=40, input_size=(65, 77, 49), num_classes=2):
        super(CNN, self).__init__()

        self.input_dim = input_dim
        
        # Define layer configs
        configs = [(input_dim, 16), (16, 32), (32, 32)]
        
        # Build conv layers from config
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 3, 1, 1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(),
                nn.MaxPool3d(2, 2),
                nn.Dropout(0.2) # 0.2 is the dropout rate
            ) for in_ch, out_ch in configs
        ])
        
        self.flatten_size = self._get_flatten_size(input_size)
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def _get_flatten_size(self, input_size):
        with torch.no_grad():
            x = torch.zeros(1, self.input_dim, *input_size)
            for layer in self.conv_layers:
                x = layer(x)
            return x.numel()
    
    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)