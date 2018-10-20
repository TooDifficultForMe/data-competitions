import torch
import torch.nn as nn
import numpy as np
from models.gcn import gcn
from torchvision import models, transforms

class model_densenet121(nn.Module):
    def __init__(self, num_class, use_gcn=False):
        super(model_densenet121, self).__init__()
        self.densenet = models.densenet121(pretrained=True)
        self.avgp = nn.AdaptiveAvgPool2d(1) if use_gcn is False else gcn(1024)
        self.fc = nn.Linear(1024, num_class)
    
    def forward(self, x):
        x = self.densenet.features(x)
        x = self.avgp(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x