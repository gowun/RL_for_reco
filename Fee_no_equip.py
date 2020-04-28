import numpy as np
import pandas as pd 
import pickle 

from mushroom_rl.environments import Environment, MDPInfo
from mushroom_rl.utils import spaces

class Fee_no_equip(Environment):
    def __init__(self, fb_labels, gamma, horizon, min_value, max_value, state_mu_sigma_abs_path, trans_model_abs_path, reward_model_abs_path):
        # MDP parameters
        self.fb_labels = fb_labels   ## fee block
        self.gamma = gamma    ## discount factor
        self.horizon = horizon    ## time limit to long
        
        self.state_mu_sigma = pickle.load(open(state_mu_sigma_abs_path, 'rb'))
        self.n_state_dim = len(self.state_mu_sigma)
        self.trans_model = pickle.load(open(trans_model_abs_path, 'rb'))  ## state transition model
        self.reward_model = pickle.load(open(reward_model_abs_path, 'rb'))

        ## ug, scores of fb classes
        self.min_point = np.ones(self.n_state_dim) * min_value
        self.max_point = np.ones(self.n_state_dim) * max_value

        self.fee_base = np.array(list(map(lambda x: float(x[3]), self.fb_labels)))
        self.max_reward = 10.0
        self._discrete_actions = list(range(len(self.fb_labels)))

        # MDP properties
        observation_space = spaces.Box(low=self.min_point, high=self.max_point)
        action_space = spaces.Discrete(len(self.fb_labels))
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        super().__init__(mdp_info)

    def reset(self, state=None):
        if state is None:
            np.random.seed()
            self._state = np.array(list(map(lambda x: np.random.normal(x[0], x[1]), self.state_mu_sigma)))
        else:
            self._state = np.array(state)
        return self._state

    def _find_next_state(self, action_onehot, ith_next):
        s = np.concatenate([self._state, action_onehot])
        return self.trans_model[ith_next].predict([s])[0]
        
    def _find_reward(self, action_onehot, next_state):
        s = np.concatenate([self._state, action_onehot, next_state])
        return self.reward_model.predict([s])[0]

    def step(self, action):
        action_onehot = np.array([False] * len(self.fb_labels))
        action_onehot[action] = True
        next_state = np.array(list(map(lambda x: self._find_next_state(action_onehot, x), range(self.n_state_dim))))
        reward = self._find_reward(action_onehot, next_state)
        
        return next_state, reward, False, {}