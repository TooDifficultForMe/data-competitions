import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms

class model_resnet34(nn.Module):
    def __init__(self, num_class):
        super(model_resnet34, self).__init__()
        self.resnet = models.resnet34(pretrained=True)
        self.argp = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_class)
    
    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.argp(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x