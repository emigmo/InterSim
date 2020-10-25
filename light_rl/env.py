import torch
import torch.nn as nn
import torch.optim as optim
import time
import serial
import os
import sys

from utils import *


class TrafficAgent():
    def __init__(self):        
        self.time0 = time.time()
        self.eff_main_cost = 2.0
        self.eff_add_cost = 0.5
        self.period_time = 80.0
        self.ser = serial.Serial("COM7", 19200, timeout=5)
        self.ser.parity = 'E'
        #self.embedding = nn.Embedding(2, 24)
        self.embedding = embed48#nn.Embedding(2, 48)
        self.embedding2 = embed1#nn.Embedding(2, 1) 
        self.light_state = np.array([1, 0])   # init light state
        self.last_reward = 0

    def get_reward(self, action,add_flag=False):
        ### the main cost value
        '''
        fh = open(COST_FILE, 'r') 
        main_cost =fh.read()
        while(main_cost == ''):
            main_cost = fh.read()
        fh.close()
        main_cost = float(main_cost)
        '''
        
        _,vel_pos = self.get_vehicle_position()
        a, b = np.ones(48), np.zeros(48)
        v1 = np.concatenate((a,b,b,b,a,b,b,b),axis=0)
        v2 = np.concatenate((b,b,a,b,b,b,a,b),axis=0)
        c1 = np.float(np.dot(vel_pos.reshape(1,-1), v1))
        c2 = np.float(np.dot(vel_pos.reshape(1,-1), v2))
        #main_cost=0.7 * c1 + 0.3 * c2

        if((self.light_state==[0,1]).all()):
            if(action[0]==1):
                main_cost=c1
            if(action[1]==1):
                main_cost=c2               
        if((self.light_state==[1,0]).all()):
            if(action[0]==1):
                main_cost=c2
            if(action[1]==1):
                main_cost=c1  



        '''
        fh = open(REWARD_FILE, 'r') 
        cost =fh.read()
        while(cost == ''):
            cost = fh.read()
        fh.close()
        main_cost = 0 - (float(cost)-self.last_reward)*10
        self.last_reward=float(cost)
        '''

        ### an additional cost value
        add_cost = 0
        if add_flag:
        ### TODO: add cost
            if((self.light_state==[0,1]).all()):
                if(action[0]==1):
                    if(c1>4):
                        add_cost+=20                        
            if((self.light_state==[1,0]).all()):
                if(action[1]==1):
                    if((c2<4)&((time.time()-self.time0)>25)):
                        add_cost+=20   
        ### compute the total cost value
        total_cost = self.eff_main_cost * main_cost + self.eff_add_cost * add_cost
        reward = np.array(-total_cost).astype(np.float32)
        reward = torch.from_numpy(reward).unsqueeze(0)
        return reward 

    def traffic_light(self):
        #### get the traffic light info
        fL=open(TRAFFIC_FILE, 'r')
        temp=fL.read()
        while(temp==''):
            temp = fL.read()
        light=np.zeros(2)
        light[0]=temp[0]
        light[1]=temp[1]
        fL.close()
        return light 

    def sync(self):
        #### syncronizing light period
        sign = self.traffic_light()
        light = self.traffic_light()
        print('Syncronizing light period...')
        while((sign == light).all()):
            fL = open(TRAFFIC_FILE, 'r')
            temp = fL.read()
            while(temp==''):
                temp = fL.read()
            light=np.zeros(2)
            light[0]=temp[0]
            light[1]=temp[1]
        self.time0 = time.time()
        self.light_state = light
        fL.close()
    
    def get_vehicle_position(self):
        while(True):  
            vel_pos = []
            ori_pos = []
            for k in list(range(1, 9)):
                fL = open(os.path.join(VEL_POSTION_DIR, 'log_direction{}.txt'.format(str(k))), 'r')
                while(True):
                    temp = fL.readlines()
                    if (np.size(temp)!=0): 
                        break
                fL.close()
                ori=temp.copy()
                l = len(temp)
                for i in range(l): 
                    temp[i] = temp[i].strip() 
                    temp[i] = temp[i].strip('[]') 
                    temp[i] = temp[i].split(",")
                    ori[i] = temp[i]
                    temp[i] = self.embedding2(torch.LongTensor([np.array(temp[i]).astype(np.float32)]))
                #vel_pos.append(np.array(temp).astype(np.float32).reshape(1,48))
                vel_pos.append(np.array(temp).reshape(48))
                ori_pos.append(np.array(ori).reshape(48))
            vel_pos = np.array(vel_pos).astype(np.float32)
            ori_pos = np.array(ori_pos).astype(np.float32)
            if vel_pos.shape==(8,48):
                fL.close()
                break       
        return vel_pos,ori_pos

    def get_light_info(self):
        '''
        light = self.traffic_light()
        if((light != self.light_state).all()):
            self.time0 = time.time()
            self.light_state = light
        light = self.light_state[0]

        N9FirstHalf = self.embedding(torch.LongTensor([light]))  # light = 0 or 1
        longOflight = time.time() - self.time0
        N9SecondHalf = torch.FloatTensor(np.ones([1, 24]) * longOflight / self.period_time)

        Node9_feat = torch.cat((N9FirstHalf, N9SecondHalf), dim=1)
        return Node9_feat
        '''
        light = self.traffic_light()
        if((light != self.light_state).all()):
            self.time0 = time.time()
            self.light_state = light
        light = self.light_state[0]
        Node9_feat = self.embedding(torch.LongTensor([light]))  # light = 0 or 1
        
        longOflight = time.time() - self.time0
        Node10_feat = torch.FloatTensor(np.ones([1, 48]) * longOflight / self.period_time)
        
        return Node9_feat, Node10_feat


    def send_action(self):
        begin = time.time()
        while ((time.time()-begin)<0.5):
            self.ser.write('1\0'.encode('gbk'))
        
    def get_state(self):
        vel_pos,_ = self.get_vehicle_position()
        light_info1, light_info2 = self.get_light_info()
        state = torch.cat((torch.FloatTensor(vel_pos), light_info1, light_info2), dim=0)
        return state

    def step(self, action):
        reward = self.get_reward(action)
        if(action[0]==1):
            self.send_action()
        time.sleep(3)    
        next_state = self.get_state()        

        return reward, next_state

    def __del__(self):
        self.ser.close()

        
if __name__=='__main__':
    pass
