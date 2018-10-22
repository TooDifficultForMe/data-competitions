import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
import pretrainedmodels

class model_seresnet(nn.Module):
    def __init__(self, num_classes):
        super(model_seresnet, self).__init__()
        self.senet = pretrainedmodels.se_resnet152()
        self.avgp = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)
        
    def forward(self, x):
        x = self.senet.features(x)
        x = self.avgp(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x