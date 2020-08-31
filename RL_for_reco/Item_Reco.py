import numpy as np
import pandas as pd 
import pickle 
import torch.nn as nn
from itertools import chain
from sklearn.ensemble import RandomForestClassifier

from mushroom_rl.environments import Environment, MDPInfo
from mushroom_rl.utils import spaces

from RL_for_reco.TorchModel import ModelMaker, FlexibleTorchModel

class Item_Reco(Environment):
    def __init__(self, items, gamma, horizon, trans_model_abs_path, item_dist=None):
        # MDP parameters

        # 1) discrete actions: list of item names or representing integers
        # 2) actions on n-dimensional space: list of a pair of min and max values per action
        self.items = items
        self.action_dim = len(self.items)
        if item_dist is None:
            if len(self.items.shape) == 1:
                if 'none' in self.items:
                    self.item_dist = np.zeros(self.action_dim)
                    self.item_dist[1:] = 1/(self.action_dim-1)
                else:
                    self.item_dist = 1/(self.action_dim)
            else:
                self.item_dist = None
        else:
            self.item_dist = item_dist
        self.gamma = gamma    ## discount factor
        self.horizon = horizon    ## time limit to long
        self.trans_model = ModelMaker(FlexibleTorchModel, model_dir_path=trans_model_abs_path)
        self.trans_model_params = self.trans_model.model.state_dict()
        tmp = list(self.trans_model_params.keys())
        key = list(filter(lambda x: '0.weight' in x, tmp))[0]
        self.state_dim = self.trans_model_params[key].shape[1] - self.action_dim
        if 'none' in self.items:
            self.state_dim += 1

        MM_VAL = 100
        self.min_point = np.ones(self.state_dim) * -MM_VAL
        self.max_point = np.ones(self.state_dim) * MM_VAL
        
        if len(self.items.shape) == 1:
            self._discrete_actions = list(range(self.action_dim))
        else:
            self._discrete_actions = None

        # MDP properties
        observation_space = spaces.Box(low=self.min_point, high=self.max_point)
        if len(self.items.shape) == 1:
            action_space = spaces.Discrete(self.action_dim)
        else:
            action_space = spaces.Box(low=self.items[0][0], high=self.items[0][1])
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        super().__init__(mdp_info)

    def reset(self, state=None):
        if state is None:
            self._state = np.zeros(self.state_dim)
        else:
            self._state = np.array(state)
        return self._state

    def step(self, action):
        if self._discrete_actions is None:
            next_state, reward = self.trans_model.infer(np.concatenate([self._state, action]))
        else:
            if 'none' in self.items:
                action_onehot = np.zeros(self.action_dim-1)
                if action > 0:
                    action_onehot[action-1] = 1.0
            else:
                action_onehot = np.zeros(self.action_dim)
                action_onehot[action] = 1.0
            next_state, reward = self.trans_model.infer(np.concatenate([self._state, action_onehot]))
        
        return next_state, reward, False, {}


def predict_actions(agent, states, items, none_tree, n_jobs=None, labeled=True):
        actions = list(map(lambda x: agent.draw_action(x), np.array(states)))
        actions = np.array(list(chain(*actions)))
        if labeled:
            str_actions = np.array(items)[actions] 
            if 'none' in str_actions:
                none_idx = np.array(range(len(str_actions)))[str_actions == 'none']

                if type(none_tree) == str:
                    try:
                        none_tree_md = pickle.load(open(none_tree, 'rb'))
                    except:
                        rec_idx = np.array(range(len(str_actions)))[str_actions != 'none']
                        none_tree_md = RandomForestClassifier(n_jobs=n_jobs, n_estimators=50, class_weight='balanced', max_features=0.8, max_depth=5, criterion='entropy').fit(np.array(states)[rec_idx], str_actions[rec_idx])
                        pickle.dump(none_tree_md, open(none_tree, 'wb'), 4)
                else:
                    none_tree_md = none_tree

                print_df = pd.DataFrame([pd.value_counts(str_actions).to_dict()])
                none_mapped = none_tree_md.predict(np.array(states)[none_idx])
                str_actions[none_idx] = none_mapped
                print_df = pd.concat([print_df, pd.DataFrame([pd.value_counts(str_actions).to_dict()])])
                print(print_df)
                return str_actions
            else:
                return str_actions
        else:
            return actions