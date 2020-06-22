import numpy as np
import pandas as pd 
import pickle 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.approximators.parametric.torch_approximator import TorchApproximator
from mushroom_rl.utils.torch import zero_grad

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
                #print(f'{self.cuda_num} is launced')
            else:
                self.network.cuda()
        if self._dropout:
            self.network.eval()

        if optimizer is not None:
            self._optimizer = optimizer['class'](self.network.parameters(),
                                                 **optimizer['params'])
        
        self._loss = loss

    def _compute_batch_loss(self, batch, use_weights, kwargs):
        if use_weights:
            weights = torch.from_numpy(batch[-1]).type(torch.float)
            if self._use_cuda:
                if self.cuda_num is None:
                    weights = weights.cuda()
                else:
                    weights = weights.to(torch.device(self.cuda_num))
            batch = batch[:-1]

        if not self._use_cuda:
            torch_args = [torch.from_numpy(x) for x in batch]
        else:
            if self.cuda_num is None:
                torch_args = [torch.from_numpy(x).cuda() for x in batch]
            else:
                torch_args = [torch.from_numpy(x).to(torch.device(self.cuda_num)) for x in batch]

        x = torch_args[:-self._n_fit_targets]

        y_hat = self.network(*x, **kwargs)

        if isinstance(y_hat, tuple):
            output_type = y_hat[0].dtype
        else:
            output_type = y_hat.dtype

        y = [y_i.clone().detach().requires_grad_(False).type(output_type) for y_i
             in torch_args[-self._n_fit_targets:]]

        if self._use_cuda:
            if self.cuda_num is None:
                y = [y_i.cuda() for y_i in y]
            else:
                y = [y_i.to(torch.device(self.cuda_num)) for y_i in y]

        if not use_weights:
            loss = self._loss(y_hat, *y)
        else:
            loss = self._loss(y_hat, *y, reduction='none')
            loss @= weights
            loss = loss / weights.sum()

        return loss

    def set_weights(self, weights):
        """
        Setter.
        Args:
            w (np.ndarray): the set of weights to set.
        """
        if self.cuda_num is None:
            set_weights(self.network.parameters(), weights, self._use_cuda)
        else:
            set_weights(self.network.parameters(), weights, self._use_cuda, self.cuda_num)

    def predict(self, *args, output_tensor=False, **kwargs):
        """
        Predict.
        Args:
            args (list): input;
            output_tensor (bool, False): whether to return the output as tensor
                or not;
            **kwargs (dict): other parameters used by the predict method
                the regressor.
        Returns:
            The predictions of the model.
        """
        if not self._use_cuda:
            torch_args = [torch.from_numpy(x) if isinstance(x, np.ndarray) else x
                          for x in args]
            val = self.network.forward(*torch_args, **kwargs)

            if output_tensor:
                return val
            elif isinstance(val, tuple):
                val = tuple([x.detach().numpy() for x in val])
            else:
                val = val.detach().numpy()
        else:
            if self.cuda_num is None:
                torch_args = [torch.from_numpy(x).cuda()
                            if isinstance(x, np.ndarray) else x.cuda() for x in args]
            else:
                torch_args = [torch.from_numpy(x).to(torch.device(self.cuda_num))
                            if isinstance(x, np.ndarray) else x.to(torch.device(self.cuda_num)) for x in args]
            val = self.network.forward(*torch_args,
                                       **kwargs)

            if output_tensor:
                return val
            elif isinstance(val, tuple):
                val = tuple([x.detach().cpu().numpy() for x in val])
            else:
                val = val.detach().cpu().numpy()

        return val

    def diff(self, *args, **kwargs):
        """
        Compute the derivative of the output w.r.t. ``state``, and ``action``
        if provided.
        Args:
            state (np.ndarray): the state;
            action (np.ndarray, None): the action.
        Returns:
            The derivative of the output w.r.t. ``state``, and ``action``
            if provided.
        """
        if not self._use_cuda:
            torch_args = [torch.from_numpy(np.atleast_2d(x)) for x in args]
        else:
            if self.cuda_num is None:
                torch_args = [torch.from_numpy(np.atleast_2d(x)).cuda() for x in args]
            else:
                torch_args = [torch.from_numpy(np.atleast_2d(x)).to(torch.device(self.cuda_num)) for x in args]

        y_hat = self.network(*torch_args, **kwargs)
        n_outs = 1 if len(y_hat.shape) == 0 else y_hat.shape[-1]
        y_hat = y_hat.view(-1, n_outs)

        gradients = list()
        for i in range(y_hat.shape[1]):
            zero_grad(self.network.parameters())
            y_hat[:, i].backward(retain_graph=True)

            gradient = list()
            for p in self.network.parameters():
                g = p.grad.data.detach().cpu().numpy()
                gradient.append(g.flatten())

            g = np.concatenate(gradient, 0)

            gradients.append(g)

        g = np.stack(gradients, -1)

        return g

def set_weights(parameters, weights, use_cuda, cuda_num=None):
    """
    Function used to set the value of a set of torch parameters given a
    vector of values.
    Args:
        parameters (list): list of parameters to be considered;
        weights (numpy.ndarray): array of the new values for
            the parameters;
        use_cuda (bool): whether the parameters are cuda tensors or not;
    """
    idx = 0
    for p in parameters:
        shape = p.data.shape

        c = 1
        for s in shape:
            c *= s

        w = np.reshape(weights[idx:idx + c], shape)

        if not use_cuda:
            w_tensor = torch.from_numpy(w).type(p.data.dtype)
        else:
            if cuda_num is None:
                w_tensor = torch.from_numpy(w).type(p.data.dtype).cuda()
            else:
                w_tensor = torch.from_numpy(w).type(p.data.dtype).to(torch.device(cuda_num))

        p.data = w_tensor
        idx += c

    assert idx == weights.size