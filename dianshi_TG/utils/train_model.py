import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
import matplotlib.pyplot as plt

class _base_trainer(object):
    def __init__(self, model=None, epoch=50, train_loader=None, test_loader=None,
                optim=None, loss_func=None, model_name=None, save=True):
        self.train_loss_rec, self.train_acc_rec = [], []
        self.test_loss_rec, self.test_acc_rec = [], []
        self.best_acc = 0
        self.model = model
        self.epoch = epoch
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optim = optim
        self.loss_func = loss_func
        self.model_name = model_name
        self.save = save
        
    def train(self):
        raise NotImplementedError
    
    def plot_loss(self):
        plt.plot(self.train_loss_rec, 'b')
        plt.plot(self.test_loss_rec, 'r')

        plt.title('loss')
        plt.xlabel('Epoch')

        plt.legend(['Train loss', 'Validation loss'], loc=4)
        plt.show()

    def plot_acc(self):
        plt.plot(self.train_acc_rec, 'b')
        plt.plot(self.test_acc_rec, 'r')

        plt.title('Prediction Accuracy')
        plt.xlabel('Epoch')

        plt.legend(['Train prediction accuracy' ,'Validation prediction accuracy'], loc=4)
        plt.show()


class trainer(_base_trainer):
    def __init__(self, model=None, epoch=50, train_loader=None, test_loader=None,
                optim=None, loss_func=None, model_name=None, save=True):
        super(trainer, self).__init__(model, epoch, train_loader, test_loader, optim,
                                     loss_func, model_name, save)
    
    def train(self):
        for epoch in range(self.epoch):
            epoch_loss = 0
            total, correct = 0, 0
            for batch_idx, (X_train, y_train) in enumerate(self.train_loader):
                X_train, y_train = X_train.to('cuda'), y_train.to('cuda')
                self.optim.zero_grad()
                y_hat = self.model(X_train)
                loss = self.loss_func(y_hat, torch.max(y_train, 1)[1])
                loss.backward()
                _, pred = torch.max(y_hat.data, 1)
                self.optim.step()
                correct += (pred == torch.max(y_train, 1)[1]).sum().item()
                total += X_train.size(0)
                epoch_loss += loss.item() / len(self.train_loader)
                print('\rEpoch {} | Batch # {} Train Loss {:.5f} '.format(epoch, batch_idx, loss.item()))
            print('\rEpoch {} | Epoch Train Loss {:.5f}'.format(epoch, epoch_loss))
            epoch_acc = correct / total * 100
            print('\nEpoch {} | Epoch Train Acc {:.3f}%'.format(epoch, epoch_acc))
            self.train_loss_rec.append(epoch_loss)
            self.train_acc_rec.append(epoch_acc)
            with torch.no_grad():
                test_epoch_loss = 0
                test_total = 0
                test_correct = 0
            
                for batch_idx, (X_val, y_val) in enumerate(self.test_loader):
                    X_val, y_val = X_val.to('cuda'), y_val.to('cuda')
            
                    y_hat = self.model(X_val)
                    loss = self.loss_func(y_hat, torch.max(y_val, 1)[1])
                    _, pred = torch.max(y_hat.data, 1)
                    test_total += y_val.size(0)
                    test_correct += (pred == torch.max(y_val, 1)[1]).sum().item()
                    test_epoch_loss += loss.item() / len(self.test_loader)
                    test_epoch_acc = test_correct / test_total * 100
                if test_epoch_acc > self.best_acc:
                    self.best_acc = test_epoch_acc
                    if self.save:
                        torch.save(self.model.state_dict(), 'weights/' + self.model_name + '/best_params_acc{}.pth'.format(best_acc)) 
                print('Epoch {} | Epoch Val Loss {:.5f}'.format(epoch, test_epoch_loss))
                print('Epoch {} | Epoch Val Acc {:.3f}%'.format(epoch, test_epoch_acc))
            
                self.test_loss_rec.append(test_epoch_loss)
                self.test_acc_rec.append(test_epoch_acc)

class cyc_trainer(_base_trainer):
    def __init__(self, model=None, epoch_per_cycle=10, cycle=6, cycle_add=1.5, train_loader=None, 
                 test_loader=None, optim=None, loss_func=None, model_name=None, save=True,
                init_lr=0.001):
        super(cyc_trainer, self).__init__(model, epoch_per_cycle, train_loader, test_loader, optim,
                                     loss_func, model_name, save)
        self.n_cycle = cycle
        self.cycle_add = cycle_add
        self.lr_list = []
        self.optim.state_dict()['param_groups'][0]['lr'] = init_lr
        self.epoch_per_cycle = epoch_per_cycle
        self.init_lr = init_lr
    
    def _cos_annealing_lr(self, initial_lr, cur_epoch, epoch_per_cycle):
        return initial_lr * (np.cos(np.pi * cur_epoch / epoch_per_cycle) + 1) / 2
    
    def train(self):
        for cycle in range(self.n_cycle):
            print('Snapshot# ', cycle)
            self.epoch_per_cycle += self.cycle_add
            for epoch in range(int(self.epoch_per_cycle)):
                epoch_loss = 0
                total, correct = 0, 0
                lr = self._cos_annealing_lr(self.init_lr, epoch, self.epoch_per_cycle)
                self.optim.state_dict()['param_groups'][0]['lr'] = lr
                self.lr_list.append(lr)
                print('\rCycle {} | Epoch {} | lr = {}'.format(cycle, epoch, lr))
                for batch_idx, (X_train, y_train) in enumerate(self.train_loader):
                    X_train, y_train = X_train.to('cuda'), y_train.to('cuda')
                    self.optim.zero_grad()
                    y_hat = self.model(X_train)
                    loss = self.loss_func(y_hat, torch.max(y_train, 1)[1])
                    loss.backward()
                    _, pred = torch.max(y_hat.data, 1)
                    self.optim.step()
                    correct += (pred == torch.max(y_train, 1)[1]).sum().item()
                    total += X_train.size(0)
                    epoch_loss += loss.item() / len(self.train_loader)
                    print('\rCycle {} | Epoch {} | Batch # {} Train Loss {:.5f} '.format(cycle, epoch, batch_idx, loss.item()))
                print('\rCycle {} | Epoch {} | Epoch Train Loss {:.5f}'.format(cycle, epoch, epoch_loss))
                epoch_acc = correct / total * 100
                print('\nCycle {} | Epoch {} | Epoch Train Acc {:.3f}%'.format(cycle, epoch, epoch_acc))
                self.train_loss_rec.append(epoch_loss)
                self.train_acc_rec.append(epoch_acc)
                with torch.no_grad():
                    test_epoch_loss = 0
                    test_total = 0
                    test_correct = 0
            
                    for batch_idx, (X_val, y_val) in enumerate(self.test_loader):
                        X_val, y_val = X_val.to('cuda'), y_val.to('cuda')
            
                        y_hat = self.model(X_val)
                        loss = self.loss_func(y_hat, torch.max(y_val, 1)[1])
                        _, pred = torch.max(y_hat.data, 1)
                        test_total += y_val.size(0)
                        test_correct += (pred == torch.max(y_val, 1)[1]).sum().item()
                        test_epoch_loss += loss.item() / len(self.test_loader)
                    test_epoch_acc = test_correct / test_total * 100
                    if test_epoch_acc > self.best_acc:
                        self.best_acc = test_epoch_acc
                        if self.save:
                            torch.save(self.model.state_dict(), 'weights/' + self.model_name + '/best_params_acc{}.pth'.format(best_acc)) 
                    print('Cycle {} | Epoch {} | Epoch Val Loss {:.5f}'.format(cycle, epoch, test_epoch_loss))
                    print('Cycle {} | Epoch {} | Epoch Val Acc {:.3f}%'.format(cycle, epoch, test_epoch_acc))
            
                    self.test_loss_rec.append(test_epoch_loss)
                    self.test_acc_rec.append(test_epoch_acc)
            self.cycle_add *= 1.5
    
    def plot_lr(self):
        plt.plot(self.lr_list)
        plt.xlabel('epoch')
        plt.ylabel('lr')
        plt.show()
    