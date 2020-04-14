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
        self.epsilon = None

    def initialise_agent(self):
        self.epsilon = ExponentialParameter(value=1, exp=0.5, size=self.env.info.observation_space.size)
        pi = EpsGreedy(epsilon=self.epsilon)

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
            if len(current_states) > 0:
                actions = list(chain(*map(lambda x: self.agent.draw_action(np.array([st])), range(len(current_states)))))
                action_arr[current_states.T[0]] = np.array(actions)

        return action_arr

    def get_Qs(self, state_names=None, action_names=None):
        if state_names is None and action_names is None:
            s_lst = list(map(lambda x: 's'+str(x), range(self.env.info.observation_space.size[0])))
            a_lst = list(map(lambda x: 'a'+str(x), range(self.env.info.action_space.size[0])))
        else:
            s_lst = state_names
            a_lst = action_names
        return pd.DataFrame(self.agent.Q.table, columns=a_lst, index=s_lst)


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
        tmp = list(map(lambda x: x.epsilon.get_value(), self.agents))
        self.mean_epsilons = np.mean(np.array(tmp), axis=1)

    def _find_best_policy(self, action_scale, action_arr, rnd):
        if rnd:
            return np.random.choice(action_arr, 1)[0]
        else:
            dist = pd.value_counts(action_arr)
            max_v = dist[0]
            print(list(zip(dist.keys(), dist.values())))
            top_actions = list(filter(lambda x: x[1] == max_v, zip(dist.keys(), dist.values())))
            
            return np.random.choice(np.array(top_actions).T[0], 1)[0]

    def draw_action_matrix(self, state_list):
        np.random.seed()
        action_matrix = pd.DataFrame({'current_state': state_list})
        for i in range(self.nEpisode):
            action_matrix[str(i) +'th_policy'] = self.agents[i].draw_action_array(state_list)
        
        action_scale = list(range(self.agents[0].P.shape[1]))
        seq_state_tuples = list(zip(range(len(state_list)), state_list))
        best_arr = np.array([-1] * len(state_list))
        for st, ep in enumerate(self.mean_epsilons):
            current_states = np.array(list(filter(lambda x: x[1] == st, seq_state_tuples)))
            if len(current_states) > 0:
                rnd_tf = np.random.choice([False, True], len(current_states), p=[ep, 1-ep])
                idxs = current_states.T[0]
                actions = list(map(lambda x: self._find_best_policy(action_scale, x[0][1:], x[1]), zip(action_matrix.values[idxs], rnd_tf)))
                best_arr[idxs] = np.array(actions)

        action_matrix['best_policy'] = best_arr
        
        return action_matrix

    def get_Qs_list(self, state_names=None, action_names=None):
        return list(map(lambda x: x.get_Qs(state_names, action_names), self.agents))