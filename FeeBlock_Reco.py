import numpy as np
import pandas as pd 
import pickle 

from mushroom_rl.environments import Environment, MDPInfo
from mushroom_rl.utils import spaces

from RL_for_reco.TorchModel import ModelMaker, FlexibleTorchModel

class FeeBlock_Reco(Environment):
    def __init__(self, fb_labels, gamma, horizon, trans_model_abs_path):
        # MDP parameters
        self.fb_labels = fb_labels   ## fee block
        self.action_dim = len(self.fb_labels)
        self.gamma = gamma    ## discount factor
        self.horizon = horizon    ## time limit to long
        self.trans_model = ModelMaker(FlexibleTorchModel, model_path=trans_model_abs_path)
        self.trans_model_params = self.trans_model.model.state_dict()
        tmp = list(self.trans_model_params.keys())
        key = list(filter(lambda x: '0.weight' in x, tmp))[0]
        self.state_dim = self.trans_model_params[key].shape[1] - self.action_dim

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