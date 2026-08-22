import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    padding_val = 1
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=padding_val, bias=False
    )


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16): 
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False), 
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(), 
        )

    def forward(self, x):
        batch_dim, channel_dim, spatial_h, spatial_w = x.size() 
        pooled_output = self.avg_pool(x).view(batch_dim, channel_dim) 
        attention_scores = self.fc(pooled_output).view(batch_dim, channel_dim, 1, 1)
        return x * attention_scores.expand_as(x)