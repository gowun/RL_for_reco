import numpy as np
import pandas as pd 
import torch
from itertools import chain
from sklearn.neighbors import NearestNeighbors

from mushroom_rl.algorithms.value import DQN, DoubleDQN, AveragedDQN
from mushroom_rl.core import Core
from mushroom_rl.environments import *
from mushroom_rl.policy import EpsGreedy, TorchPolicy
from mushroom_rl.approximators.parametric.torch_approximator import TorchApproximator
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.utils.parameters import Parameter, LinearParameter

from RL_for_reco.FeeBlock_Reco import FeeBlock_Reco
from RL_for_reco.Network_for_Reco import Network_for_Reco

ALG_NAMES = {'DQN': DQN, 'DDQN': DoubleDQN, 'ADQN': AveragedDQN}
PI_NAMES = {'EG': EpsGreedy, 'TP': TorchPolicy}
ENV_NAMES = {'FBR': FeeBlock_Reco}

class DQN_Learn:
    def __init__(self, env_name, pi_name, alg_name, env_params={}, pi_params={}, alg_params={}):
        ## MDP
        self.env_name = ENV_NAMES[env_name]
        self.env = self.env_name(**env_params)

        ## Policy
        self.pi_name = PI_NAMES[pi_name]
        self.cuda_tf = True if torch.cuda.is_available() else False
        if pi_name == 'EG':
            if len(pi_params) == 1 and list(pi_params.keys()) == ['value']:
                epsilon = Parameter(**pi_params)
            else:
                epsilon = LinearParameter(**pi_params)
            self.policy = EpsGreedy(epsilon=epsilon)
        else:
            self.policy = TorchPolicy(self.cuda_tf)

        ## Parameters of Network_for_Reco
        self.alg_params = alg_params
        self.alg_params['use_cuda'] = self.cuda_tf
        self.alg_params['network'] = Network_for_Reco
        self.alg_params['input_shape'] = self.env.info.observation_space.shape
        self.alg_params['output_shape'] = self.env.info.action_space.size
        self.alg_params['n_actions'] = self.env.info.action_space.n

        ## Parameters of Agent
        self.agent_params = {}
        for key in ['batch_size', 'target_update_frequency', 'initial_replay_size', 'max_replay_size', 'n_approximators']:
            '''
            batch_size (int): the number of samples in a batch;
            target_update_frequency (int): the number of samples collected
                between each update of the target network;
            initial_replay_size (int, 500): the number of samples to collect before
                starting the learning;
            max_replay_size (int, 5000): the maximum number of samples in the replay
                memory;
            n_approximators (int, 1): the number of approximator to use in
                ``AveragedDQN``;
            '''
            try:
                self.agent_params[key] = self.alg_params[key]
                del self.alg_params[key]
            except KeyError:
                pass
        self.agent_params['mdp_info'] = self.env.info
        self.agent_params['policy'] = self.policy
        self.agent_params['approximator'] = TorchApproximator
        self.agent_params['approximator_params'] = self.alg_params

        ## Agent and Core
        self.alg_name = ALG_NAMES[alg_name]
        self.agent = self.alg_name(**self.agent_params)
        self.core = Core(self.agent, self.env)

    def train(self, n_epochs, n_steps, train_frequency):
        for _ in range(n_epochs):
            self.core.learn(n_steps=n_steps, n_steps_per_fit=train_frequency)

    def compare_model_with_origin(self, initial_states, compared_rewards, n_samples=10000):
        if len(initial_states) > n_samples:
            idx = np.random.choice(range(len(initial_states)), n_samples, replace=False)
            samples = initial_states[idx]
            if compared_rewards is not None:
                raw_r = np.mean(np.array(compared_rewards)[idx])
        else:
            samples = initial_states
            if compared_rewards is not None:
                raw_r = np.mean(compared_rewards)
        dataset = self.core.evaluate(initial_states=samples)
        J = compute_J(dataset, 1.0)
        learned_r = np.mean(J)/self.env.horizon
        return learned_r, raw_r, learned_r - raw_r

    def draw_actions(self, states, labeled=True):
        actions = list(map(lambda x: self.agent.draw_action(np.array([x])), np.array(states)))
        actions = np.array(list(chain(*actions)))
        if labeled:
            if self.env_name == FeeBlock_Reco:
                str_actions = np.array(self.env.fb_labels)[actions]
                if 'none' in str_actions:
                    none_idx = np.array(range(len(actions)))[str_actions == 'none']
                    knn = NearestNeighbors(10).fit(states)
                    neighbors = list(map(lambda x: knn.kneighbors(states[x], 10, return_distance=False), none_idx))
                    most_frq = list(map(lambda x: list(pd.value_counts(x).keys())[0], neighbors))
                    str_actions[none_idx] = np.array(most_frq)

                return str_actions
        else:
            return actions