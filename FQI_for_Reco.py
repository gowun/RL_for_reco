import numpy as np
import pandas as pd 
from sklearn.ensemble import ExtraTreesRegressor

from mushroom_rl.algorithms.value import FQI
from mushroom_rl.core import Core
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.parameters import Parameter
from mushroom_rl.utils.dataset import compute_J

### Environment
from RL_for_reco.Fee_no_equip import Fee_no_equip
ENV = {'FNE': Fee_no_equip}

class FQI_for_Reco:
    def __init__(self, env_name, exp, **kwarg):
        np.random.seed()

        # MDP
        self.env_name = ENV[env_name]
        self.env = self.env_name(**kwarg)

        self.exp = exp     ## learning rate for epsilong greedy
        self.agent = None
        self.core = None

    def initialise_all(self, n_iter):
        # Policy
        learning_rate = Parameter(value=1.0)
        pi = EpsGreedy(epsilon=learning_rate)   ## the policy followed by the agent

        # Approximator
        approximator_params = dict(input_shape=self.env.info.observation_space.shape, 
                                    n_actions=self.env.info.action_space.n,
                                    n_estimators=50,
                                    min_samples_split=5,
                                    min_samples_leaf=3)
        approximator = ExtraTreesRegressor
        algorithm_params = dict(n_iterations=n_iter)
        self.agent = FQI(self.env.info, pi, approximator, approximator_params=approximator_params, **algorithm_params)
        self.core = Core(self.agent, self.env)

    def learn_agent(self, n_episodes):
        self.core.learn(n_episodes=n_episodes, n_episodes_per_fit=n_episodes)

    def draw_str_actions(self, states):
        actions = list(map(lambda x: self.core.agent.draw_action(np.array([x]))[0], states))
        
        return actions

    def evaluate_given_states(self, test_df):
        init_states = test_df[self.env.next_cols].values

        dataset = self.core.evaluate(initial_states=init_states)
        mean_rewards = np.mean(compute_J(dataset, self.env.info.gamma))

        return dataset, mean_rewards