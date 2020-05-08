import numpy as np
import pandas as pd 
import pickle 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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
            q_acted = torch.squeeze(q.gather(1, action))

            return q_acted