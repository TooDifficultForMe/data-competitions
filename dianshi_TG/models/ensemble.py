import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from data import data_reader

class ensemble(nn.Module):
    def __init__(self, num_models, num_features, num_classes):
        super(ensemble, self).__init__()
        assert type(num_features) is list
        assert num_models == len(num_features)
        self.fc_list = [nn.Sequential(nn.Linear(i, 512), nn.ReLU(inplace=True)) for i in num_features]
        self.tail = nn.Sequential(nn.Linear(num_models * 512, 512), 
                                  nn.ReLU(inplace=True), 
                                  nn.Linear(512, num_classes))
    def forward(self, feature_list):
        assert type(feature_list) is list
        assert len(feature_list) == len(self.fc_list)
        y = [fc(feature) for fc, feature in zip(self.fc_list, feature_list)]
        y = torch.cat(y, -1)
        return self.tail(y)
    