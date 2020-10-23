import copy
import torch 
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, SpGraphAttentionLayer
import numpy as np
import pdb

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1: 4])
        fan_out = np.prod(weight_shape[2: 4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0.0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1.0)
        m.bias.data.fill_(0.0)
    elif classname.find('LSTMCell') != -1:
        m.bias_ih.data.fill_(0.0)
        m.bias_hh.data.fill_(0.0)

class GATmodel(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GATmodel, self).__init__()
        self.dropout = dropout
        self.number_of_actions = 2
        self.gamma = 0.99
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.001
        self.number_of_iterations = 10000
        self.replay_memory_size = 5000
        self.minibatch_size = 50
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        self.mix_mlp = nn.Linear(16, 2)

        self.apply(weights_init)  # init weight

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))

        return x 
        # mix=self.mix_mlp(x.mean(dim=0))
        # mix=F.softmax(mix,dim=0)
        # return mix


class end_layer(nn.Module):
    def __init__(self, in_channels=32, out_channels=1):
        super(end_layer, self).__init__()
        self.fc1 = nn.Linear(in_channels, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, out_channels)
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(16)

        self.apply(weights_init)

    def forward(self, x):
        if x.size()[0] == 1:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        else:
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.fc2(x)))
        return self.fc3(x)

class GATActorCritic(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, num_actions=2):
        super(GATActorCritic, self).__init__()
        self.dropout = dropout
        self.number_of_actions = 2
        self.gamma = 0.99
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.001
        self.number_of_iterations = 10000
        self.replay_memory_size = 5000
        self.minibatch_size = 50

        self.num_actions = num_actions
        self.feat = GATmodel(nfeat, nhid, nclass, dropout, alpha, nheads)
        self.critic = end_layer(in_channels=nclass, out_channels=1)
        self.action = end_layer(in_channels=nclass, out_channels=num_actions)

        self.apply(weights_init)  # init weight
        self.train()
    def forward(self, x, adj):
        x = self.feat.forward(x, adj).mean(dim=0).unsqueeze(0)#####
        return self.critic(x), self.action(x)
