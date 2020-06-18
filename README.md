# Deep Reinforcement Learning for Business Structured Data
---

## Item_Reco
---

A class to recommend products to customers with their any current information and product-recommended history.
Class variable items indicates the products as well as their associate promotions, offers such as any recommendation type. 
If you want to take a case where customers have not recommendation, you can use 'none' to represent the case.
States, actions and reward are respectively n-dim array, 1-d array and a float number.
A transition model, state + action => (state, reward), is assumed as a multi-output neural network on TorchModel. 

This framework, actually, is applicable to problems of any structured data.



## Network_for_Reco
---

A class to update Q-values though a nueral network.
This is also a general form avaiable to any problem.


## DQN_Learn
---

A class to formulate a Deep Q Learning problem(an environment, an agent and its policy and associated parameters) and to learn the agent by a Deep Q Network and its approximator. 


## TorchModel
---

Several classes to build a neural network by pyTorch.