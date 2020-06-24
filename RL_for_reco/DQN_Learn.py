import numpy as np
import pandas as pd 
import torch
import pickle

from mushroom_rl.algorithms.value import DQN, DoubleDQN, AveragedDQN
from mushroom_rl.core import Core
from mushroom_rl.environments import *
from mushroom_rl.policy import EpsGreedy, Boltzmann#, TorchPolicy
from mushroom_rl.approximators.parametric.torch_approximator import TorchApproximator
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.utils.parameters import Parameter, LinearParameter, ExponentialParameter

from RL_for_reco.Item_Reco import Item_Reco
from RL_for_reco.Network_for_Reco import Network_for_Reco, TorchApproximator_cuda

ALG_NAMES = {'DQN': DQN, 'DDQN': DoubleDQN, 'ADQN': AveragedDQN}
PI_PR_NAMES = {'Static': Parameter, 'Linear': LinearParameter, 'Exp': ExponentialParameter}
PI_NAMES = {'EG': EpsGreedy, 'BTM': Boltzmann}
ENV_NAMES = {'IR': Item_Reco}

class DQN_Learn:
    def __init__(self, env_name, pi_pr_name, pi_name, alg_name, env_params={}, pi_pr_params={}, alg_params={}):
        ## MDP
        self.env_name = ENV_NAMES[env_name]
        self.env = self.env_name(**env_params)

        ## Policy
        self.pi_pr_name = PI_PR_NAMES[pi_pr_name]
        self.pi_name = PI_NAMES[pi_name]
        self.policy = self.pi_name(self.pi_pr_name(**pi_pr_params.copy()))

        ## Parameters of Network_for_Reco
        self.alg_params = alg_params.copy()
        #self.alg_params['use_cuda'] = True if torch.cuda.is_available() else False
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
        self.agent_params['approximator'] = TorchApproximator#_cuda
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
