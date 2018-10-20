import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms

class model_densenet121(nn.Module):
    def __init__(self, num_class):
        super(model_densenet121, self).__init__()
        self.densenet = models.densenet121(pretrained=True)
        self.avgp = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_class)
    
    def forward(self, x):
        x = self.densenet.features(x)
        x = self.avgp(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x