import numpy as np
import pandas as pd 
import pickle 

from mushroom_rl.environments import Environment, MDPInfo
from mushroom_rl.utils import spaces

class Fee_no_equip(Environment):
    def __init__(self, ug_labels, fb_labels, gamma, horizon, sas_abs_path, knn_model_abs_path, ug_model_abs_path):
        # MDP parameters
        self.ug_labels = ug_labels   ## user group
        self.fb_labels = fb_labels   ## fee block
        self.gamma = gamma    ## discount factor
        self.horizon = horizon    ## time limit to long
        """
        sas_dataset: state-action-state, pandas dataframe
        - state: cluster membership + score of actions
        - action: fee block of price plans according to their fees

        state / given action / next state
        """
        self.sas_dataset = pickle.load(open(sas_abs_path, 'rb'))
        self.next_cols = list(filter(lambda x: 'next' in x, self.sas_dataset.columns))
        self.knn_model = pickle.load(open(knn_model_abs_path, 'rb')) ## ug 별로 존재, two components of model and mapping from associate to original indexes
        self.ug_model = pickle.load(open(ug_model_abs_path, 'rb'))  ## clustering
        #self.score_model = pickle.load(open(score_model_abs_path, 'rb'))  ## multi-classification 

        ## ug, scores of fb classes
        self.min_point = np.zeros(1 + len(self.fb_labels))
        self.max_point = np.array([float(len(self.ug_labels)-1)] + [1.0] * len(self.fb_labels))

        self.fee_base = np.array(list(map(lambda x: float(x[3]), self.fb_labels)))
        self.max_reward = 10.0

        # MDP properties
        observation_space = spaces.Box(low=self.min_point, high=self.max_point)
        action_space = spaces.Discrete(len(self.fb_labels))
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        super().__init__(mdp_info)

    def reset(self, state=None):
        if state is None:
            np.random.seed()
            state_former = np.random.choice(self.ug_labels)
            state_latter = np.random.random_sample(len(self.fb_labels))
            self._state = np.concate([[float(state_former)], state_latter])
        else:
            self._state = state

    def _estimate_current_state(self, model_idx, a):
        idxs = self.knn_model[model_idx]['model'].kneighbors([self._state[1:]], self.knn_model[model_idx]['size'], return_distance=False)[0]
        idxs = list(map(lambda x: self.knn_model[model_idx]['mapping'][x], idxs))
        return self.sas_dataset.iloc[idxs].query(f"rec_{a} > 0.0")

    def step(self, action):
        ### find the most similar case in sas_dataset
        model_idx = int(self._state[0])
        a = self.fb_labels[action]
        pos_states = self._estimate_current_state(model_idx, a)
        if len(pos_states) > 0:
            next_state = pos_states[self.next_cols].values[0]
        else:
            for cl in self.ug_model['order'][model_idx]:
                pos_states = self._estimate_current_state(cl, a)
                if len(pos_states) > 0:
                    next_state = pos_states[self.next_cols].values[0]
                    break

        fa = float(a[3])
        """
        weight rule
        1) 가장 비싼 요금제로 이동, max_reward
        2) 추천한 요금제 이상의 요금제로 이동, 요금제 최대값 + 1, 즉 7
        3) 추천한 요금제 미만의 요금제로 이동, 이동한 요금제 - 추천한 요금제
        """
        weights = list(map(lambda x: self.max_reward if x == max(self.fee_base) else (max(self.fee_base) + 1 if x >= fa else x - fa), self.fee_base))
        ## -1.8 < reward < 1.9
        reward = sum(next_state[1:] * np.array(weights)) ** (1/len(self.fb_labels))
        
        return next_state, reward, False, {}