import pickle
import time
import numpy as np
import pandas as pd
from itertools import chain
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TorchDataCreation:
    def __init__(self, *data_list):
        self.data_list = [np.array(d).astype(float) for d in data_list]
        self.n_data = len(self.data_list)
        self.data_length = len(self.data_list[0])
    
    def split_index(self, test_ratio):
        self.test_idx = np.random.choice(self.data_length, round(self.data_length*test_ratio), replace=False)
        self.train_idx = np.array(list(set(range(self.data_length)) - set(self.test_idx)))
    
    def get_dataset(self, idx=None):
        torchdata_list = []
        for d in self.data_list:
            if idx is None:
                tmp = d
            else:
                tmp = d[idx]
            torchdata_list.append(torch.Tensor(tmp).float())
        return TensorDataset(*torchdata_list)
        
    def get_dataloader(self, torchdata, **params):
        return DataLoader(dataset=torchdata, **params)
    
    def make_dataloader(self, **params):
        self.train_loader = self.get_dataloader(self.get_dataset(self.train_idx), **params)
        params['shuffle'] = False
        self.test_loader = self.get_dataloader(self.get_dataset(self.test_idx), **params)
        

class FlexibleTorchModel(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dims, drop_rate=0.0):
        super(FlexibleTorchModel, self).__init__()
        self.drop_rate = drop_rate
        
        self.fully_connected_net = []
        in_size = in_dim
        for i, next_size in enumerate(hidden_dims):
            fc = nn.Linear(in_features=in_size, out_features=next_size)
            in_size = next_size
            self.__setattr__('fc{}'.format(i), fc)
            self.fully_connected_net.append(fc)
            
        self.last_layers = []
        for i, od in enumerate(out_dims):
            last = nn.Linear(in_features=in_size, out_features=od)
            self.__setattr__('last{}'.format(i), last)
            self.last_layers.append(last)

    def forward(self, X):
        for i, fc in enumerate(self.fully_connected_net):
            X = F.dropout(F.relu(fc(X)), p=self.drop_rate)
        outputs = []
        for last in self.last_layers:
            outputs.append(last(X))
        
        return outputs
    


class ModelMaker:
    def __init__(self, model, model_path=None, **model_params):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model
        if model_path is None:
            self.model_params = model_params
            self.model = self.model_name(**self.model_params).to(self.device)
        else:
            self.load_model(model_path)
        
    def load_model(self, model_path):
        tmp = pickle.load(open(model_path, 'rb'))
        self.model_params = tmp[1]
        self.model = self.model_name(**self.model_params).to('cpu')
        self.model.load_state_dict(tmp[0])
    
    @torch.no_grad()
    def infer(self, any_data):
        if type(any_data) != torch.Tensor:
            tensor_data = torch.tensor(torch.tensor(np.array(any_data).astype(float)).float())
    
        self.model.eval()
        tmp = self.model.forward(tensor_data)
        inferred = []
        for ifr in tmp:
            ifr = ifr.numpy()
            if len(ifr.shape) == 2 and ifr.shape[1] == 1:
                ifr = np.array(list(chain(*ifr)))
            inferred.append(ifr)
    
        return inferred
    
    def evaluate_model_byMSE(self, data_set):
        pred = self.infer(data_set[0])
        mse_list = []
        for i, y in enumerate(data_set[1:]):
            if len(y.shape) == 1:
                mse_list.append(mean_squared_error(y, pred[i]))
            else:
                mse_list.append(np.mean(list(map(lambda x: mean_squared_error(x[0], x[1]), zip(np.array(y), pred[i])))))
        return mse_list
    
    def set_optimizer(self, lr_rate, w_decay=1e-8):
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr_rate, weight_decay=w_decay)
        
    def set_criterions(self, *metrics):
        self.criterions = metrics
    
    def single_run(self, tensor_datasets, train=False, loss_p=None):
        outputs = self.model(tensor_datasets[0])
                
        losses = []
        for i, out in enumerate(outputs):
            if len(self.criterions) == 1:
                ls = self.criterions[0](out, tensor_datasets[i+1])
            else:
                ls = self.criterions[i](out, tensor_datasets[i+1])
            losses.append(ls)
        if loss_p is None:
            loss = torch.add(*losses)
        else:
            loss = sum(np.array(losses) * np.array(loss_p))
    
        if train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item()
    
    def print_process(self, i, total_iter, batch_time, loss, epoch=0, train=True):
        if train:
            print('Epoch {0} [{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch + 1, i + 1, total_iter, 
                                                                  batch_time=batch_time, loss=loss))
        else:
            print('Test [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i + 1, total_iter, 
                                                                  batch_time=batch_time, loss=loss))

    def train(self, loader, epoch, loss_p=None, print_term=None):
        batch_time = AverageMeter()
        losses = AverageMeter()
        
        self.model.train()
    
        total_iter = len(loader)
        end = time.time()
        
        for i, data in enumerate(loader):
            data_tensor = [ts.to(self.device) for ts in data]
            
            loss = self.single_run(data_tensor, True, loss_p)
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            batch_size = data_tensor[0].size(0)
            losses.update(loss, batch_size)
            
            if print_term is not None and i % print_term == 0:
                self.print_process(i, total_iter, batch_time, losses, epoch, True)
        print('***\t Loss {loss.avg:.4f}'.format(loss=losses))
        
    @torch.no_grad()
    def validate(self, loader, loss_p=None, print_term=None):
        batch_time = AverageMeter()
        losses = AverageMeter()
        
        self.model.eval()
    
        total_iter = len(loader)
        end = time.time()
        
        for i, data in enumerate(loader):
            data_tensor = [ts.to(self.device) for ts in data]
            
            loss = self.single_run(data_tensor, False, loss_p)
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            batch_size = data_tensor[0].size(0)
            losses.update(loss, batch_size)
            
            if print_term is not None and i % print_term == 0:
                self.print_process(i, total_iter, batch_time, losses, 0, False)
        print('***\t Loss {loss.avg:.4f}'.format(loss=losses))
    
    def train_validate(self, train_loader, test_loader, total_epoch, loss_p=None, print_term=None):
        for ep in range(total_epoch):
            self.train(train_loader, ep, loss_p, print_term)
            self.validate(test_loader, loss_p, print_term)
    
    def save_model(self, abs_path):
        pickle.dump([self.model.state_dict(), self.model_params], open(abs_path, 'wb'), 4)
        #torch.save(self.model.state_dict(), abs_path)