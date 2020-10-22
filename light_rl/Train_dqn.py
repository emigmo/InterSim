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
model = GATmodel(nfeat=48, nhid=32, nclass=16, dropout=0.6, alpha=0, nheads=1)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = nn.MSELoss().cuda()
AdjacentMatrix=torch.Tensor(ADJ_matrix)
edge = AdjacentMatrix.to(device) 

replay_memory = []

env = TrafficAgent()
env.sync()

epsilon_decrements = np.linspace(model.initial_epsilon, model.final_epsilon, model.number_of_iterations)
epsilon = model.initial_epsilon
iteration = 0
state = env.get_state()
ax = []                    
ay = []
while iteration < model.number_of_iterations:
    output = model.forward(state.to(device), edge)       
    # generate the action [light change or not]
    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    if torch.cuda.is_available():
        action = action.cuda()
    if random.random() <= epsilon:
        action_index = torch.randint(model.number_of_actions, torch.Size([]), dtype=torch.int)
        print("Performed random action !")
    else:
        action_index = torch.argmax(output.squeeze(0))
    action[action_index] = 1

    #  execute action into UE4
    reward, next_state = env.step(action)
    epsilon = epsilon_decrements[iteration]

    replay_memory.append((state, action, reward, next_state))
    if len(replay_memory) > model.replay_memory_size:
        replay_memory.pop(0)
    minibatch = random.sample(replay_memory, min(len(replay_memory), model.minibatch_size))

    state_batch = torch.cat(tuple(d[0].unsqueeze(0) for d in minibatch)).to(device)
    action_batch= torch.cat(tuple(d[1].unsqueeze(0) for d in minibatch)).to(device)
    reward_batch = torch.cat(tuple(d[2] for d in minibatch)).to(device)
    state_l_batch = torch.cat(tuple(d[3].unsqueeze(0) for d in minibatch)).to(device)
    
    output_l_batch = torch.cat(tuple(model(state_l_batch[d],edge).unsqueeze(0) for d in range(0,min(len(replay_memory), model.minibatch_size))),dim=0)

    y_batch = torch.cat(tuple(reward_batch[i].unsqueeze(0) + model.gamma * torch.max(output_l_batch[i].unsqueeze(0))
                                for i in range(len(minibatch))))

    q_value = torch.sum(torch.cat(tuple(model(state_batch[d],edge).unsqueeze(0) for d in range(0,min(len(replay_memory), model.minibatch_size)))) * action_batch, dim=1)
    #####################################
    globals()['ax'].append(iteration)
    globals()['ay'].append(-reward)
    plt.clf()
    plt.plot(ax,ay)
    plt.pause(0.1)
    plt.ioff()
    #####################################
    optimizer.zero_grad()
    y_batch = y_batch.detach()
    loss = criterion(q_value, y_batch)        
    loss.backward(retain_graph=True )
    optimizer.step()

    iteration += 1
    state = next_state

    if iteration % 1000 == 0:
        torch.save(replay_memory, "GAT+RL/pretrained_model/replay_memory" + str(iteration) + ".pth")
        torch.save(model, "GAT+RL/pretrained_model/current_model_" + str(iteration) + ".pth")
        np.savetxt('iteration'+str(iteration)+'.txt',globals()['ax'],fmt='%0.8f')
        np.savetxt('cost'+str(iteration)+'.txt',globals()['ay'],fmt='%0.8f')           
        plt.savefig(str(iteration)+".png")
    print("iteration:", iteration, "light time:", time.time() - env.time0,  "action:",
            action_index.cpu().detach().numpy(), "COST:",-reward.numpy() )
