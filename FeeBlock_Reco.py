import numpy as np
import pandas as pd 
import pickle 
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed

from mushroom_rl.environments import Environment, MDPInfo
from mushroom_rl.utils import spaces

from RL_for_reco.TorchModel import ModelMaker, FlexibleTorchModel

class FeeBlock_Reco(Environment):
    def __init__(self, fb_labels, gamma, horizon, trans_model_abs_path, fb_dist=None):
        # MDP parameters
        self.fb_labels = fb_labels   ## fee block
        self.action_dim = len(self.fb_labels)
        if fb_dist is None:
            self.fb_dist = np.zeros(self.action_dim)
            self.fb_dist[1:] = 1/(self.action_dim-1)
        else:
            self.fb_dist = fb_dist
        self.gamma = gamma    ## discount factor
        self.horizon = horizon    ## time limit to long
        self.trans_model = ModelMaker(FlexibleTorchModel, model_path=trans_model_abs_path)
        self.trans_model_params = self.trans_model.model.state_dict()
        tmp = list(self.trans_model_params.keys())
        key = list(filter(lambda x: '0.weight' in x, tmp))[0]
        self.state_dim = self.trans_model_params[key].shape[1] - self.action_dim + 1

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
        action_onehot = np.zeros(self.action_dim-1)
        action_onehot[action-1] = 1.0
        next_state, reward = self.trans_model.infer(np.concatenate([self._state, action_onehot]))
        
        return next_state, reward, False, {}


def approximate_none(states, str_actions, cols, n_neighbors, n_jobs):
    new_actions = str_actions.copy()
    print_df = pd.DataFrame([pd.value_counts(str_actions).to_dict()], columns=cols)
    none_idx = np.array(range(len(str_actions)))[str_actions == 'none']
    act_idx = np.array(range(len(str_actions)))[str_actions != 'none']
    n_neighbors = min(len(act_idx), n_neighbors)
    knn = NearestNeighbors(n_neighbors, n_jobs=n_jobs).fit(states[act_idx])
    neighbors = knn.kneighbors(states[none_idx], n_neighbors, return_distance=False)
    nei_actions = list(map(lambda x: str_actions[x], neighbors))
    most_frq = Parallel(n_jobs=n_jobs)(delayed(find_most_frq)(x) for x in nei_actions)
    #most_frq = list(map(lambda x: self._find_most_frq(x), nei_actions))
    new_actions[none_idx] = np.array(most_frq)
    print_df = pd.concat([print_df, pd.DataFrame([pd.value_counts(str_actions).to_dict()])])
    print(print_df)

    return new_actions, str_actions


def find_most_frq(self, lst, ignore=['none']):
    tmp = pd.value_counts(lst)
    if self.env_name == FeeBlock_Reco:
        action_space = self.env.fb_labels
        action_dist = self.env.fb_dist

    if len(tmp) == 1:
        if list(tmp.keys())[0] in ignore:
            return np.random.choice(action_space, 1, p=action_dist)
        else:
            return list(tmp.keys())[0]
    else:
        action_scores = np.zeros(len(action_space))
        for i, a in enumerate(action_space):
            if a in tmp.keys():
                action_scores[i] = tmp[a] * action_dist[i]
        return action_space[np.argmax(action_scores)]
