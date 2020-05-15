import numpy as np
import pandas as pd 
import torch
import torch.nn as nn
from itertools import chain

from RL_for_reco.FeeBlock_Reco import FeeBlock_Reco, approximate_none
import RL_for_reco.TorchModel as tm

ENV_NAMES = {'FBR': FeeBlock_Reco}

class CrossEntropy_Learn:
    def __init__(self, env_name, loss_name, cuda_num=None, lr=0.005, env_params={}, net_params={}):
        self.env_name = ENV_NAMES[env_name]
        self.env = self.env_name(**env_params)
        if cuda_num is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(f'cuda: {cuda_num}')

        self.network = tm.ModelMaker(tm.FlexibleTorchModel, cuda_num, **net_params)
        self.network.set_optimizer(lr)
        self.loss_name = loss_name
        if self.loss_name == 'crossentropy':
            self.network.set_criterions(nn.CrossEntropyLoss())
        elif self.loss_name == 'focal':
            self.network.set_criterions(self.network.__focal_loss())
        elif self.loss_name == 'mse':
            self.network.set_criterions(nn.MSELoss())

    def generate_random_episode(self, n_steps):
        state_long = []
        action_long = []
        reward = 0.0
        sm = nn.Softmax(dim=1)
        state = self.env.reset()
        for _ in range(n_steps):
            action_probs = sm(self.network(torch.FloatTensor([state]))).data.numpy()[0]
            action = np.random.choice(len(self.env.action_dim), p=action_probs)
            if self.loss_name in ['crossentropy', 'focal']:
                action_long.append(action)
            else:
                action_onehot = np.zeros(self.env.action_dim)
                action_onehot[action] = 1.0
                action_long.append(action_onehot)
            state_long.append(state)
            action_long.append(action)
            reward += self.env.step(action)[1]

            state = self.env.reset(state)

        return state_long, action_long, reward

    def generate_episode_batch(self, batch_size, min_step=100, max_step=1000):
        batch = []

        for _ in range(batch_size):
            n_steps = np.random.choice(range(min_step, max_step))
            batch.append(self.generate_random_episode(n_steps))

        return batch

    def filter_batch(self, batch, percentile):
        rewards = list(map(lambda x: x[2], batch))
        reward_bound = np.percentile(rewards, percentile)
        reward_mean = np.mean(rewards)

        train_obs = []
        train_act = []
        for episode in batch:
            if episode[2] >= reward_bound:
                train_obs.append(episode[0])
                train_act.append(episode[1])

        train_obs = torch.FloatTensor(train_obs)
        if self.loss_name in ['crossentropy', 'focal']:
            train_act = torch.LongTensor(train_act)
        else:
            train_act = torch.FloatTensor(train_act)

        return train_obs, train_act, reward_bound, reward_mean

    def train(self, percentile, batch_size, min_step=100, max_step=1000, delta=0.1):
        result = []
        for i, batch in enumerate(self.generate_episode_batch(batch_size, min_step, max_step)):
            obs_v, act_v, rw_b, rw_m = self.filter_batch(batch, percentile)

            data_tensor = [v.to(self.device) for v in [obs_v, act_v]]

            self.network.optimizer.zero_grad()
            action_scores = self.network.model(data_tensor[0])
            if self.loss_name == 'focal':
                loss_v = self.network.criterions[0](action_scores[0].cpu(), data_tensor[1].cpu())
            else:
                loss_v = self.network.criterions[0](action_scores[0], data_tensor[1])
            loss_v.backward()
            self.network.optimizer.step()

            result.append([i, loss_v.item(), rw_m, rw_b])
            print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % result[-1])

            if i > 0 and abs(result[-1][2] - result[-2][2]) < delta:
                break
            i += 1
        
        return result
    
    def draw_actions(self, states, n_neighbors=100, n_jobs=20):
        action_probs = self.network.infer(states)[0]
        raw_actions = np.array(self.env.fb_labels)[np.argmax(action_probs, axis=0)]

        if 'none' in raw_actions:
            return approximate_none(states, raw_actions, self.env.fb_labels, self.env.action_dist, n_neighbors, n_jobs)
        else:
            return raw_actions