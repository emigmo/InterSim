import numpy as np

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

COST_FILE = 'D:\cost.txt'
TRAFFIC_FILE = 'D:\TrafficLight.txt'
VEL_POSTION_DIR = 'D:\DataRaw\DataSet\'