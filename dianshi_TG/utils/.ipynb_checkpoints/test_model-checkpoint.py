import pandas as pd
from data import data_reader, TGDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from torchvision import models, transforms
import pretrainedmodels

class submitter(object):
    def __init__(self, model=None, im_trm=None, test_df=None,
                 class_list=['CITY', 'DESERT', 'FARMLAND', 'LAKE', 'MOUNTAIN', 'OCEAN'],
                path=None):
        assert model.training is False, 'the model must be in eval mode!'
        self.model = model
        self.im_trm = im_trm
        self.class_list = class_list
        self.test_df = test_df
        self.path = path
        
    def _im_read(self, path):
        img = Image.open(path)
        return img.convert('RGB')
        
    def to_csv(self, csv_name='submission.csv'):
        sub = pd.DataFrame()
        for i in range(len(self.test_df)):
            im = self._im_read(self.path + self.test_df[0][i])
            x = self.model(self.im_trm(im).unsqueeze(0))
            c = torch.max(x, 1)[1].item()
            sub = sub.append([(self.test_df[0][i], self.class_list[c])])
            if i % 100 == 0: print(i)
        sub.to_csv(csv_name, header=None, index=None, encoding='utf-8')