import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
import pretrainedmodels

class model_pnasnet(nn.Module):
    def __init__(self, num_class):
        super(model_pnasnet, self).__init__()
        self.pnas = pretrainedmodels.pnasnet5large(num_classes=1000)
        self.argp = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4320, num_class)
    
    def forward(self, x):
        x = self.pnas.features(x)
        x = self.argp(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x