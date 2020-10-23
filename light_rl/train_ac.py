import torch
import torch.nn as nn
import torch.optim as optim
import time
import random

import matplotlib.pyplot as plt

import os
import sys
import copy
import pdb

from utils import *
from gat import GATmodel
from env import TrafficAgent

device = torch.device("cuda:0")
# build model
model = GATActorCritic(nfeat=48, nhid=32, nclass=16, dropout=0.2, alpha=0.2, nheads=2, num_actions=2)
model = model.to(device)

if not os.path.exists('GAT+RL/pretrained_model/'):
    os.mkdir('GAT+RL/pretrained_model/')

optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

AdjacentMatrix=torch.Tensor(ADJ_matrix)
edge = AdjacentMatrix.to(device) 

replay_memory = []
env = TrafficAgent()
env.sync()

# epsilon_decrements = np.linspace(model.initial_epsilon, model.final_epsilon, model.number_of_iterations)
# epsilon = model.initial_epsilon
iteration = 0
state = env.get_state()
ax = []                    
ay = []

ac_num_steps = 120
gamma = 0.99
tau = 1.0
entropy_coef = 0.01
value_loss_coef = 0.5
max_grad_norm = 20.0
number_of_iterations = 10000
while iteration < number_of_iterations:
    training = True
    episode_length = 0

    while training:
        values = []
        log_probs = []
        rewards = []
        entropies = []

        for step in range(ac_num_steps):  # 120 timestep as one trajectory
            iteration += 1
            episode_length += 1

            value, logit = model.forward(state.to(device), edge)       
            prob = F.softmax(logit, dim=1)
            log_prob = F.log_softmax(logit, dim=1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)   # push entropy
            
            action_index = prob.multinomial(num_samples=1).data.cpu()   # action sample from actor policy

            action = torch.zeros([model.number_of_actions], dtype=torch.float32)
            action[action_index] = 1
            if torch.cuda.is_available():
                _action = action.cuda()
            log_prob = log_prob.gather(1, _action)

            #  execute action into UE4
            reward, next_state = env.step(action)
            #TODO plot reward value.

            state = next_state  # s' --> s
            
            values.append(value)         # push value 
            log_probs.append(log_prob)   # push log_prob
            rewards.append(reward)       # reward

        R, _ = model(state.to(device), edge)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        if torch.cuda.is_available():
            R = R.cuda()
            gae = gae.cuda()
        values.append(R)
        for i in reversed(range(len(rewards))):
            R = gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = rewards[i] + gamma * values[i + 1].data - values[i].data
            gae = gae * gamma * tau + delta_t

            policy_loss = policy_loss - log_probs[i] * Variable(gae) - entropy_coef * entropies[i]

        optimizer.zero_grad()

        (policy_loss + value_loss_coef * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()