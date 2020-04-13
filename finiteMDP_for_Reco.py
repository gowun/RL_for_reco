import numpy as np
import pandas as pd
from itertools import chain
from mushroom_rl.environments.finite_mdp import FiniteMDP
from mushroom_rl.algorithms.value import QLearning, DoubleQLearning, WeightedQLearning, SpeedyQLearning, SARSA
from mushroom_rl.core import Core
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.callbacks import CollectDataset, CollectMaxQ
from mushroom_rl.utils.dataset import parse_dataset
from mushroom_rl.utils.parameters import ExponentialParameter

ALGORITHM_DICT = {'QL': QLearning, 'DQL': DoubleQLearning, 'WQL': WeightedQLearning, 'SQL': SpeedyQLearning, 'SARSA': SARSA}

class FMDP_Reco:
    def __init__(self, algorithm, exp, P, R):
        self.agentAlgorithm = ALGORITHM_DICT[algorithm]
        self.exp = exp
        self.P = P
        self.R = R
        self.env = FiniteMDP(P, R)
        self.agent = None
        self.core = None

    def initialise_agent(self):
        epsilon = ExponentialParameter(value=1, exp=0.5, size=self.env.info.observation_space.size)
        pi = EpsGreedy(epsilon=epsilon)

        learning_rate = ExponentialParameter(value=1, exp=self.exp, size=self.env.info.size)
        algorithm_params = dict(learning_rate=learning_rate)
        self.agent = self.agentAlgorithm(self.env.info, pi, **algorithm_params)

        start = self.env.reset()
        collect_max_Q = CollectMaxQ(self.agent.approximator, start)
        collect_dataset = CollectDataset()
        callbacks = [collect_dataset, collect_max_Q]
        self.core = Core(self.agent, self.env, callbacks)
    
    def learn_agent(self, n_step):
        self.core.learn(n_step, n_steps_per_fit=1)

    def draw_action_array(self, state_list):
        seq_state_tuples = list(zip(range(len(state_list)), state_list))
        action_arr = np.array([-1] * len(state_list))

        for st in range(self.env.info.observation_space.size[0]):
            current_states = np.array(list(filter(lambda x: x[1] == st, seq_state_tuples)))
            actions = list(chain(*map(lambda _: self.agent.draw_action(np.array([st])), range(len(current_states)))))
            action_arr[current_states.T[0]] = np.array(actions)

        return action_arr


class Multiple_FMDP_Reco:
    def __init__(self, nEpisode, nStep, algorithm_list, exp, P, R):
        self.nEpisode = nEpisode
        self.nStep = nStep
        self.agents = []
        for i in range(self.nEpisode):
            np.random.seed()
            self.agents.append(FMDP_Reco(algorithm_list[i], exp, P, R))
            self.agents[-1].initialise_agent()
            self.agents[-1].learn_agent(self.nStep)

    def draw_action_matrix(self, state_list):
        np.random.seed()
        action_matrix = pd.DataFrame({'current_state': state_list})
        for i in range(self.nEpisode):
            action_matrix[str(i) +'th_policy'] = self.agents[i].draw_action_array(state_list)

        action_matrix['best_policy'] = list(map(lambda x: pd.value_counts(x[1:]).keys()[0], action_matrix.values))
        
        return action_matrix