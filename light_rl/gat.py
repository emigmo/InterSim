import copy
import torch 
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, SpGraphAttentionLayer
import pdb

class TwoLayerMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TwoLayerMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc_1 = nn.Linear(input_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.LeakyReLU(0.2)  # the same activation as GAT paper
    def forward(self, x):
        hidden = self.activation(self.fc_1(x))
        output = self.fc_2(hidden)
        return output
class CombineMLP(nn.Module):
    def __init__(self, self_embed_dim, neighbor_dim, hidden_dim, output_dim):
        super(CombineMLP, self).__init__()
        self.fc_1 = nn.Linear(self_embed_dim + neighbor_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.LeakyReLU(0.2)
    def forward(self, self_embed, sum_neighbor):
        x = torch.cat((self_embed, sum_neighbor), dim=-1)
        hidden = self.activation(self.fc_1(x))
        output = self.fc_2(hidden)
        return output

class GAT(nn.Module):
    def __init__(self, node_dim, timestep, hidden_dim, output_dim, encode_times=1):
        super(GAT, self).__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.timestep = timestep
        self.encode_times = encode_times
        input_dim = node_dim * timestep
        self.self_mlp = TwoLayerMLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.neighbor_mlp = TwoLayerMLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.combine_mlp = CombineMLP(self_embed_dim=hidden_dim, neighbor_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=1)
        self.output_mlp = nn.Linear(hidden_dim, 1)

    def forward(self, nodes, edges):  
        "input nodes shape: [batch, node, timesteps, attribute]"
        # origin_nodes = copy.deepcopy(nodes)
        batch_size = nodes.shape[0]
        num_node = nodes.shape[1]
        nodes = nodes.reshape((batch_size, num_node, -1))
        for i in range(self.encode_times):
            node_self_embed = self.self_mlp(nodes)
            node_neighbor_embed = self.neighbor_mlp(nodes)
            updated_self_embed = torch.zeros(node_self_embed.shape)
            for batch in range(batch_size):
                for node_index in range(num_node):
                    selected_neighbors_importance = []
                    selected_neighbors_embed = []
                    for other_node_index in range(num_node):
                        this_edge = edges[batch, other_node_index, node_index] # from other nodes to this node
                        if this_edge != 0 and node_index != other_node_index:
                            # selected_index.append(other_node_index)
                            selected_neighbors_embed.append(node_neighbor_embed[batch, other_node_index])
                            selected_neighbors_importance.append(self.combine_mlp(node_self_embed[batch, node_index], node_neighbor_embed[batch, other_node_index]))
                    if selected_neighbors_importance:
                        selected_neighbors_importance = torch.tensor(selected_neighbors_importance, dtype=torch.float32)
                        weights = F.softmax(selected_neighbors_importance)
                        selected_neighbors_embed = torch.stack(selected_neighbors_embed, dim=0)
                        "here, we use the same mlp to encode embeddings to be sumed and embeddings to calculate the weights"
                        sum_neighbor_embed = (weights.unsqueeze(1) * selected_neighbors_embed).sum(dim=0)
                        updated_self_embed[batch, node_index] = sum_neighbor_embed + node_self_embed[batch, node_index]
            nodes = updated_self_embed
        return updated_self_embed
        
class GATmodel(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
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
        self.mix_mlp=nn.Linear(2, 2)
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        mix=self.mix_mlp(x.mean(dim=0))
        mix=F.softmax(mix,dim=0)
        return mix