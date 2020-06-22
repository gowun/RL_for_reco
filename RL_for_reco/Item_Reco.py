import numpy as np
import pandas as pd 
import pickle 
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed

from mushroom_rl.environments import Environment, MDPInfo
from mushroom_rl.utils import spaces

from RL_for_reco.TorchModel import ModelMaker, FlexibleTorchModel

class Item_Reco(Environment):
    def __init__(self, items, gamma, horizon, trans_model_abs_path, item_dist=None):
        # MDP parameters
        self.items = items   ## fee block
        self.action_dim = len(self.items)
        if item_dist is None:
            if 'none' in self.items:
                self.item_dist = np.zeros(self.action_dim)
                self.item_dist[1:] = 1/(self.action_dim-1)
            else:
                self.item_dist = 1/(self.action_dim)
        else:
            self.item_dist = item_dist
        self.gamma = gamma    ## discount factor
        self.horizon = horizon    ## time limit to long
        self.trans_model = ModelMaker(FlexibleTorchModel, model_path=trans_model_abs_path)
        self.trans_model_params = self.trans_model.model.state_dict()
        tmp = list(self.trans_model_params.keys())
        key = list(filter(lambda x: '0.weight' in x, tmp))[0]
        self.state_dim = self.trans_model_params[key].shape[1] - self.action_dim
        if 'none' in self.items:
            self.state_dim += 1

        MM_VAL = 100
        self.min_point = np.ones(self.state_dim) * -MM_VAL
        self.max_point = np.ones(self.state_dim) * MM_VAL

        self._discrete_actions = list(range(self.action_dim))

        # MDP properties
        observation_space = spaces.Box(low=self.min_point, high=self.max_point)
        action_space = spaces.Discrete(self.action_dim)
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        super().__init__(mdp_info)

    def reset(self, state=None):
        if state is None:
            self._state = np.zeros(self.state_dim)
        else:
            self._state = np.array(state)
        return self._state

    def step(self, action):
        if 'none' in self.items:
            action_onehot = np.zeros(self.action_dim-1)
            if action > 0:
                action_onehot[action-1] = 1.0
        else:
            action_onehot = np.zeros(self.action_dim)
            action_onehot[action] = 1.0
        next_state, reward = self.trans_model.infer(np.concatenate([self._state, action_onehot]))
        
        return next_state, reward, False, {}

'''
def approximate_none(states, str_actions, action_space, action_dist, n_neighbors, n_jobs):
    new_actions = str_actions.copy()
    print_df = pd.DataFrame([pd.value_counts(str_actions).to_dict()], columns=action_space)
    none_idx = np.array(range(len(str_actions)))[str_actions == 'none']
    act_idx = np.array(range(len(str_actions)))[str_actions != 'none']
    n_neighbors = min(len(act_idx), n_neighbors)
    knn = NearestNeighbors(n_neighbors, n_jobs=n_jobs).fit(states[act_idx])
    neighbors = knn.kneighbors(states[none_idx], n_neighbors, return_distance=False)
    act_actions = str_actions[act_idx]
    nei_actions = list(map(lambda x: act_actions[x], neighbors))
    most_frq = list(map(lambda x: find_most_frq(x, action_space, action_dist), nei_actions))
    new_actions[none_idx] = np.array(most_frq)
    print_df = pd.concat([print_df, pd.DataFrame([pd.value_counts(new_actions).to_dict()])])
    print(print_df)

    return new_actions


def find_most_frq(lst, action_space, action_dist, ignore=['none']):
    tmp = list(map(lambda x: sum(x == np.array(lst)), action_space))
    action_scores = np.array(tmp) * np.array(action_dist)

    return action_space[np.argmax(action_scores)]

'''