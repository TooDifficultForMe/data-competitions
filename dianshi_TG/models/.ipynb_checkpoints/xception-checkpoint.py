import torch
import torch.nn as nn
import numpy as np
from models.gcn import gcn
from torchvision import models, transforms
import pretrainedmodels

class model_xception(nn.Module):
    def __init__(self, num_class, use_gcn=False):
        super(model_xception, self).__init__()
        self.xception = pretrainedmodels.models.xception()
        self.avgp = nn.AdaptiveAvgPool2d(1) if use_gcn is False else gcn(2048, 10)
        self.fc = nn.Linear(2048, num_class)
    
    def forward(self, x):
        x = self.xception.features(x)
        x = self.avgp(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x