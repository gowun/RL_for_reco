import numpy as np
import pandas as pd 
import pickle 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.approximators.parametric.torch_approximator import TorchApproximator

class Network_for_Reco(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_dims, **kwargs):
        super().__init__()
        self.output_size = output_shape[0]

        self.fully_connected_net = []
        in_size = input_shape[-1]
        for i, next_size in enumerate(hidden_dims):
            hh = nn.Linear(in_features=in_size, out_features=next_size)
            nn.init.xavier_uniform_(hh.weight, gain=nn.init.calculate_gain('relu'))
            in_size = next_size
            self.__setattr__(f'_h{i}', hh)
            self.fully_connected_net.append(hh)

        self.last_layer = nn.Linear(in_features=in_size, out_features=output_shape[0])
        nn.init.xavier_uniform_(self.last_layer.weight, gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action=None):
        features = torch.tensor(state).float()
        for hh in self.fully_connected_net:
            features = F.relu(hh(torch.squeeze(torch.tensor(features).float(), 1).float()))
        q = self.last_layer(torch.tensor(features).float())

        if action is None:
            return q
        else:
            action = action.long()
            q_acted = torch.squeeze(q.gather(1, action))

            return q_acted


class TorchApproximator_cuda(TorchApproximator):
    def __init__(self, input_shape, output_shape, network, optimizer=None,
                 loss=None, batch_size=0, n_fit_targets=1, use_cuda=False,
                 reinitialize=False, dropout=False, quiet=True, cuda_num=None, **params):

        self._batch_size = batch_size
        self._reinitialize = reinitialize
        self._use_cuda = use_cuda
        self._dropout = dropout
        self._quiet = quiet
        self._n_fit_targets = n_fit_targets
        self.cuda_num = None if cuda_num is None else f'cuda: {cuda_num}'

        self.network = network(input_shape, output_shape, **params)

        if self._use_cuda:
            if cuda_num is not None:
                self.network.to(torch.device(self.cuda_num))
                print(f'{self.cuda_num} is launced')
            else:
                self.network.cuda()
        if self._dropout:
            self.network.eval()

        if optimizer is not None:
            self._optimizer = optimizer['class'](self.network.parameters(),
                                                 **optimizer['params'])
        
        self._loss = loss

        