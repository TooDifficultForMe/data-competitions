import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
import pretrainedmodels
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from data import data_reader

Tensor = torch.Tensor
rand_m = np.random.random

class model_pnas(nn.Module):
    def __init__(self, num_class):
        super(model_pnas, self).__init__()
        self.pnas = pretrainedmodels.pnasnet5large(num_classes=1000)
        self.argp = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4320, num_class)
    
    def forward(self, x):
        x = self.pnas.features(x)
        x = self.argp(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
model = model_pnas(6).to('cuda')


optim = torch.optim.Adam(model.parameters())
loss_func = torch.nn.CrossEntropyLoss()

from data import data_reader
class args:
    bs = 8
    path = 'train2000/'

p_norm = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
p_trm = transforms.Compose([
    transforms.Resize(331),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    p_norm
])

val_p_trm = transforms.Compose([
    transforms.Resize(333),
    transforms.CenterCrop(331),
    transforms.ToTensor(),
    p_norm
])
    
    
dr = data_reader(args, 'train2000/train2000.csv', trm=p_trm, val_trm=val_p_trm)

train_loader, test_loader = dr.get_train_loader()

train_loss_rec = []
train_acc_rec = []
test_loss_rec = []
test_acc_rec = []
best_acc = 0
for epoch in range(50):
    epoch_loss = 0
    total, correct = 0, 0
    for batch_idx, (X_train, y_train) in enumerate(train_loader):
        X_train, y_train = X_train.to('cuda'), y_train.to('cuda')
        optim.zero_grad()
        y_hat = model(X_train)
        loss = loss_func(y_hat, torch.max(y_train, 1)[1])
        loss.backward()
        _, pred = torch.max(y_hat.data, 1)
        optim.step()
        correct += (pred == torch.max(y_train, 1)[1]).sum().item()
        total += X_train.size(0)
        epoch_loss += loss.item() / len(train_loader)
        print('\rEpoch {} | Batch # {} Train Loss {:.5f} '.format(epoch, batch_idx, loss.item()))
    print('\rEpoch {} | Epoch Train Loss {:.5f}'.format(epoch, epoch_loss))
    epoch_acc = correct / total * 100
    print('\nEpoch {} | Epoch Train Acc {:.3f}%'.format(epoch, epoch_acc))
    train_loss_rec.append(epoch_loss)
    train_acc_rec.append(epoch_acc)
    with torch.no_grad():
        test_epoch_loss = 0
        test_total = 0
        test_correct = 0
            
        for batch_idx, (X_val, y_val) in enumerate(test_loader):
            X_val, y_val = X_val.to('cuda'), y_val.to('cuda')
            
            y_hat = model(X_val)
            loss = loss_func(y_hat, torch.max(y_val, 1)[1])
            _, pred = torch.max(y_hat.data, 1)
            test_total += y_val.size(0)
            test_correct += (pred == torch.max(y_val, 1)[1]).sum().item()
            test_epoch_loss += loss.item() / len(test_loader)
             
        test_epoch_acc = test_correct / test_total * 100
        if test_epoch_acc > best_acc:
            best_acc = test_epoch_acc
            torch.save(model.state_dict(), 'weights/' + MODEL_NAME + '/best_params_acc{}.pth'.format(best_acc)) 
        print('Epoch {} | Epoch Val Loss {:.5f}'.format(epoch, test_epoch_loss))
        print('Epoch {} | Epoch Val Acc {:.3f}%'.format(epoch, test_epoch_acc))
            
        test_loss_rec.append(test_epoch_loss)
        test_acc_rec.append(test_epoch_acc)