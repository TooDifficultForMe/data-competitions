import torch
import torch.nn as nn

class gcn(nn.Module):
    def __init__(self, ch, ks=7):
        super(gcn, self).__init__()
        self.conv_l1 = nn.Conv2d(ch, ch, (ks, 1))
        self.conv_l2 = nn.Conv2d(ch, ch, (1, ks))
        self.conv_r1 = nn.Conv2d(ch, ch, (1, ks))
        self.conv_r2 = nn.Conv2d(ch, ch, (ks, 1))
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        return self.relu(x_l + x_r)