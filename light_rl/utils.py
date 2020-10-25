import numpy as np
import torch
import torch.nn as nn
'''
ADJ_matrix = np.array(
              [[1,0,0,0,0,0,0,0,1],
              [0,1,0,0,0,0,0,0,1],
              [0,0,1,0,0,0,0,0,1],
              [0,0,0,1,0,0,0,0,1],
              [0,0,0,0,1,0,0,0,1],
              [0,0,0,0,0,1,0,0,1],
              [0,0,0,0,0,0,1,0,1],
              [0,0,0,0,0,0,0,1,1],
              [1,1,1,1,1,1,1,1,1]])
'''
ADJ_matrix = np.array(
              [[1,0,0,0,0,0,0,0,1,1],
              [0,1,0,0,0,0,0,0,1,1],
              [0,0,1,0,0,0,0,0,1,1],
              [0,0,0,1,0,0,0,0,1,1],
              [0,0,0,0,1,0,0,0,1,1],
              [0,0,0,0,0,1,0,0,1,1],
              [0,0,0,0,0,0,1,0,1,1],
              [0,0,0,0,0,0,0,1,1,1],
              [1,1,1,1,1,1,1,1,1,1],
              [1,1,1,1,1,1,1,1,1,1]])
COST_FILE = 'D:\cost.txt'
TRAFFIC_FILE = 'D:\TrafficLight.txt'
VEL_POSTION_DIR = 'D:\DataRaw\DataSet'
REWARD_FILE='D://reward.txt'

embed1 = torch.load('embed1.pkl').eval()
embed48 = torch.load('embed48.pkl').eval()
