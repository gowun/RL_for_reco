import numpy as np
import pandas as pd 
import torch
import pickle

from mushroom_rl.algorithms.value import DQN, DoubleDQN, AveragedDQN
from mushroom_rl.algorithms.actor_critic import A2C, DDPG, TD3, SAC, TRPO, PPO
from mushroom_rl.core import Core
from mushroom_rl.environments import *
from mushroom_rl.policy import EpsGreedy, Boltzmann, GaussianTorchPolicy, OrnsteinUhlenbeckPolicy
from mushroom_rl.approximators.parametric.torch_approximator import TorchApproximator
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.utils.parameters import Parameter, LinearParameter, ExponentialParameter

from RL_for_reco.Item_Reco import Item_Reco
from RL_for_reco.Network_for_Reco import Network_for_Reco, TorchApproximator_cuda

ALG_NAMES = {'DQN': DQN, 'DDQN': DoubleDQN, 'ADQN': AveragedDQN,   ## value-based
             'A2C': A2C, 'DDPG': DDPG, 'TD3': TD3, 'SAC': SAC, 'TRPO': TRPO, 'PPO': PPO}   ## policy-based
PI_PR_NAMES = {'Static': Parameter, 'Linear': LinearParameter, 'Exp': ExponentialParameter}  ## only for value-based
PI_NAMES = {'EG': EpsGreedy, 'BTM': Boltzmann, 
            'GSTorch': GaussianTorchPolicy, 'OUNoise': OrnsteinUhlenbeckPolicy}
ENV_NAMES = {'IR': Item_Reco}

'''
** Policy Gradient **
- A2C, PPO: need policy model, 1 critic network and 1 actor optimizer
- TRPO: need policy model and 1 critic network with max_kl
- DDPG, TD3: need (noise) policy model, 1 critic network, 1 actor network and 1 actor optimizer
- SAC: 2 actor networks and 1 critic network and 1 actor optimizer
'''


class RL_Learn:
    def __init__(self, env_name, alg_name, pi_pr_name=None, pi_name=None, env_params={}, pi_pr_params={}, alg_params={}):
        """
        Constructor

        env_name: only IR
        alg_name: one of {'DQN': DQN, 'DDQN': DoubleDQN, 'ADQN': AveragedDQN,   ## value-based
                          'A2C': A2C, 'DDPG': DDPG, 'TD3': TD3, 'SAC': SAC, 'TRPO': TRPO, 'PPO': PPO}   ## policy-based
        pi_pr_name(default=None): one of {'Static': Parameter, 'Linear': LinearParameter, 'Exp': ExponentialParameter}
        pi_name(default=None): one of {'EG': EpsGreedy, 'BTM': Boltzmann, ## value-based
                                       'GSTorch': GaussianTorchPolicy, ## policy-based
                                       'OUNoise': OrnsteinUhlenbeckPolicy}  ## add noise
        
        env_params: 
            items, array of item labels or integers representing items for value-based learning; 
                   array of a pair of min and max value over all items(actually all the pair are the same, in this version)
            gamma, learning rate
            horizon, length for agent to long in the environment
            trans_model_abs_path, absolute directory path where a transition model from (state, action) to (state, reward) is
            item_dist(default=None), a distribution item happening; when default, all the probability is the same as 1/|items|

        pi_pr_params:
            'Static'
                value, initial value
                min_value(default=None)
                max_value(default=None)
            'Linear'
                value, initial value
                threshold_value, some value to know (threshold_value - value)/n; 
                                 if which is smaller than 0, it is to be min value else to be max value
                n, denominator
            'Exp'
                value, initial value
                exp(default=1.0), used to be the power as value/(n ** exp) where n is the current number of being trained
                min_value(default=None)
                max_value(default=None)

            pi_name='EG'
                epsilon, one of 'Static', 'Linear' and 'Exp', the exploration coefficient
            pi_name='BTM'
                beta, one of 'Static', 'Linear' and 'Exp', the inverse of the temperature distribution; 
                      As the temperature approaches 0.0, the policy becomes more and more greedy
            pi_name='GSTorch'
                std_0(default=1.0), initial standard deviation
            pi_name='OUNoise'
                sigma, array of critic network output size, average magnitude of the random fluctuation per square-root time
                theta, rate of mean reversion
                dt, time interval
                x0(default=None), array of critic network output size, initial values of noise

        alg_params:
            ## params for Network_for_Reco
            hidden_dims, list/array of hidden unit sizes
            optimizer, dict, parameters to specify the optimizer algorithm
            loss, the loss function

            ### params for agent algorithms for DQN
            alg_name in ['DQN', 'DDQN', 'ADQN']
                batch_size, the number of samples in a batch
                target_update_frequency, the number of samples collected between each update of the target network
                replay_memory(default=None), one of ReplayMemory and PrioritizedReplayMemory, the object of the replay memory to use; 
                                             if None, a default replay memory is created
                initial_replay_size(default=500), the number of samples to collect before starting the learning
                max_replay_size(default=5000), the maximum number of samples in the replay memory
                n_approximators(default=1) the number of approximator for alg_name='ADQN'
                clip_reward(default=True), whether to clip the reward or not

            ### params for agent of actor-critic
            alg_name in ['A2C', 'PPO', 'DDPG', 'TD3', 'SAC']
                alg_name='A2C'
                    ent_coeff(default=0.0), coefficient for the entropy penalty
                    max_grad_norm(default=None), maximum norm for gradient clipping; 
                                                If None, no clipping will be performed, unless specified otherwise in actor_optimizer
                alg_name='PPO'
                    n_epochs_policy, number of policy updates for every dataset
                    batch_size, size of minibatches for every optimization step
                    eps_ppo, value for probability ratio clipping
                    lam(default=1.), lambda coefficient used by generalized advantage estimation
                alg_name in ['DDPG', 'TD3]
                    batch_size, the number of samples in a batch
                    initial_replay_size, the number of samples to collect before starting the learning
                    max_replay_size, the maximum number of samples in the replay memory
                    tau, value of coefficient for soft updates
                    policy_delay(default=1), the number of updates of the critic after which an actor update is implemented; 
                                             for 'TD3', default=2
                    alg_name='TD3'
                        noise_std(default=.2), standard deviation of the noise used for policy smoothing
                        noise_clip(default=.5), maximum absolute value for policy smoothing noise
                alg_name='SAC'
                    batch_size, the number of samples in a batch
                    initial_replay_size, the number of samples to collect before starting the learning
                    max_replay_size, the maximum number of samples in the replay memory
                    warmup_transitions, number of samples to accumulate in the replay memory to start the policy fitting
                    tau, value of coefficient for soft updates
                    lr_alpha, Learning rate for the entropy coefficient
                    target_entropy(default=None),target entropy for the policy; if None a default value is computed
            alg_name='TRPO'
                ent_coeff(default=0.0), coefficient for the entropy penalty
                max_kl(default=.001), maximum kl allowed for every policy update
                lam(default=1.), lambda coefficient used by generalized advantage estimation
                n_epochs_line_search(default=10), maximum number of iterations of the line search algorithm
                n_epochs_cg(default=10), maximum number of iterations of the conjugate gradient algorithm
                cg_damping(default=1e-2), damping factor for the conjugate gradient algorithm
                cg_residual_tol(default=1e-10), conjugate gradient residual tolerance
        """

        ## MDP
        self.env_name = ENV_NAMES[env_name]
        self.env = self.env_name(**env_params)
        

        ## Parameters of Agent(agent_params) and of Network_for_Reco(alg_params)
        self.agent_params = alg_params.copy()

        self.alg_params = {}
        for key in ['hidden_dims', 'optimizer', 'loss']:
            self.alg_params[key] = alg_params[key]
            if key == 'optimizer' and alg_name in ['A2C', 'PPO', 'DDPG', 'TD3', 'SAC']:
                self.agent_params.update({'actor_optimizer': alg_params[key]})
            del self.agent_params[key]
        self.alg_params['network'] = Network_for_Reco
        self.alg_params['input_shape'] = self.env.info.observation_space.shape
        if len(self.env.items.shape) == 1:
            self.alg_params['output_shape'] = self.env.info.action_space.size
            self.alg_params['n_actions'] = self.alg_params['output_shape'][0]
            self.alg_params['mode'] = 'q'
        else:
            self.alg_params['output_shape'] = (self.env.action_dim,)
            if alg_name in ['DDPG', 'TD3', 'SAC']:
                self.alg_params['mode'] = 'actor'

                self.crt_alg_params = self.alg_params.copy()
                del self.alg_params['optimizer']
                self.crt_alg_params['mode'] = 'critic'
                self.crt_alg_params['input_shape'] = (self.env.state_dim + self.env.action_dim,)
                self.crt_alg_params['output_shape'] = (1,)
            else:
                self.alg_params['mode'] = 'critic'
                self.alg_params['output_shape'] = (1,)

        # Policy 
        if pi_name is not None:
            self.pi_name = PI_NAMES[pi_name]
            if pi_name == 'OUNoise' and pi_pr_params is not None:
                self.pi_pr_params = pi_pr_params
            elif pi_name.endswith('Torch'):
                self.pi_pr_params = {
                    'use_cuda': False,
                    'network': Network_for_Reco,
                    'input_shape': self.env.info.observation_space.shape,
                    'output_shape': (self.env.action_dim,),
                    'hidden_dims': self.alg_params['hidden_dims']
                }
                self.pi_pr_params.update(pi_pr_params)
                self.policy = self.pi_name(**self.pi_pr_params)
            elif pi_pr_name is not None:
                self.pi_pr_name = PI_PR_NAMES[pi_pr_name]
                self.policy = self.pi_name(self.pi_pr_name(**pi_pr_params.copy()))


        self.agent_params['mdp_info'] = self.env.info
        if alg_name in ['DDPG', 'TD3']:
            self.agent_params['policy_class'] = self.pi_name
            self.agent_params['policy_params'] = self.pi_pr_params
            self.agent_params['actor_params'] = self.alg_params
            self.agent_params['critic_params'] = self.crt_alg_params
        elif alg_name == 'SAC':
            self.agent_params['critic_params'] = self.crt_alg_params
            self.agent_params['actor_mu_params'] = self.alg_params.copy()
            self.agent_params['actor_sigma_params'] = self.alg_params.copy()
        else:
            if alg_name != 'TRPO':
                self.agent_params['policy'] = self.policy
            
            if alg_name in ['DQN', 'DDQN', 'ADQN']:
                self.agent_params['approximator'] = TorchApproximator#_cuda
                self.agent_params['approximator_params'] = self.alg_params
            else:
                self.agent_params['critic_params'] = self.alg_params

        ## Agent and Core
        self.alg_name = ALG_NAMES[alg_name]
        self.agent = self.alg_name(**self.agent_params)
        self.core = Core(self.agent, self.env)

    def train(self, n_epochs, n_steps, train_frequency, print_epoch=True):
        for i in range(n_epochs):
            if print_epoch:
                print(f'---------- {i}th epoch -----------')
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
