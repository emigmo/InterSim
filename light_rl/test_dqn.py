import torch
import torch.nn.functional as F
from torch.autograd import Variable
import time
import random

import matplotlib.pyplot as plt

import os
import sys
import copy
import pdb

from utils import *
from gat import GATmodel,GATActorCritic
from env import TrafficAgent

device = torch.device("cuda:0")
# Load model
model = GATmodel(nfeat=48, nhid=84, nclass=16, dropout=0.1, alpha=0.2, nheads=1)
model.load_state_dict(torch.load('GAT+RL/pretrained_model/current_model_100.pkl'))
model = model.to(device)

AdjacentMatrix=torch.Tensor(ADJ_matrix)
edge = AdjacentMatrix.to(device) 

env = TrafficAgent()
env.sync()

iteration = 0
state = env.get_state()
ax = []                    
ay = []
reward=torch.Tensor(0)#for display
while True:
    output = model.forward(state.to(device), edge)
    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    if torch.cuda.is_available():
        action = action.cuda()
    action_index = torch.argmax(output.squeeze(0))
    action[action_index] = 1

    print("iteration:", iteration, "light time:", time.time() - env.time0,  "action:",
        action_index.cpu().detach().numpy(), "COST:",-reward.numpy() )
    reward, next_state = env.step(action)
    state=next_state
    #####################################
    globals()['ax'].append(iteration)
    globals()['ay'].append(-reward)
    plt.clf()
    plt.plot(ax,ay)
    plt.pause(0.1)
    plt.ioff()
    #####################################
    iteration+=1