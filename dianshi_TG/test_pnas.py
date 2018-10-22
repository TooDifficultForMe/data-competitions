from utils.test_model import submitter
import pandas as pd
from data import data_reader, TGDataset
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
import numpy as np
from torchvision import models, transforms
import pretrainedmodels
import matplotlib.pyplot as plt

Tensor = torch.Tensor
rand_m = np.random.random
PATH = 'test1000/'

p_norm = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])

val_p_trm = transforms.Compose([
    transforms.Resize(333),
    transforms.CenterCrop(331),
    transforms.ToTensor(),
    p_norm
])
trm = val_p_trm

df = pd.read_csv('test1000.csv', header=None)
from models import model_pnasnet
pnas = model_pnasnet(6)
pnas.load_state_dict(torch.load('weights/pnasnet5large/best_params_acc94.5.pth'))
pnas.eval()

s = submitter(pnas, trm, df, path='test1000/')
s.to_csv('tes.csv')