import cv2 
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import argparse

import time
import serial
import matplotlib.pyplot as plt

import os
import sys
import csv

import copy
import torch.nn.functional as F
from layers import GraphAttentionLayer, SpGraphAttentionLayer
import pdb

device = torch.device("cuda:0")

##################################################################################################
class GAT(nn.Module):
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



def train(model, start):
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = nn.MSELoss().cuda()

    AdjacentMatrix=torch.Tensor(np.array([[1,0,0,0,0,0,0,0,1],[0,1,0,0,0,0,0,0,1],[0,0,1,0,0,0,0,0,1],[0,0,0,1,0,0,0,0,1],[0,0,0,0,1,0,0,0,1],[0,0,0,0,0,1,0,0,1],[0,0,0,0,0,0,1,0,1],[0,0,0,0,0,0,0,1,1],[1,1,1,1,1,1,1,1,1]]))
    edge=AdjacentMatrix.to(device)   
    num_node = torch.ones(1).to(device) * 9
    #############################################################################
    fh = open('D:\cost.txt', 'r') 
    cost=fh.read()
    while(cost==''):
        cost=fh.read()
    fh.close()
    fL=open('D:\TrafficLight.txt','r')
    temp=fL.read()
    light=np.zeros(2)
    light[0]=temp[0]
    light[1]=temp[1]
    fL.close()
    
    sign=light
    print('syncronizing light period')
    while((sign==light).all()):
        fL=open('D:\TrafficLight.txt','r')
        temp=fL.read()
        while(temp==''):
            temp=fL.read()
        light=np.zeros(2)
        light[0]=temp[0]
        light[1]=temp[1]
    
    sign=light
    time0=time.time()
    fL.close()

    while(True):
        for k in [1,2,3,4,5,6,7,8]:
            fL=open('D:\DataRaw\DataSet\log_direction'+str(k)+'.txt','r')
            while(True):
                temp=fL.readlines()
                if (np.size(temp)!=0): break
            fL.close()
            l = len(temp)
            for i in range(l): 
                temp[i] = temp[i].strip() 
                temp[i] = temp[i].strip('[]') 
                temp[i] = temp[i].split(",")
            globals()['pos'+str(k)]=np.array(temp).astype(np.float32).reshape(1,48)
            if(k==1):pos=pos1
            else:   
                pos=np.concatenate((pos,globals()['pos'+str(k)]),axis=0)
        if pos.shape==(8,48):
            fL.close()
            break    

    """cost2 is an additional cost induced by the number of cars remained on roads
    """
    a=np.ones(48)
    b=np.zeros(48)
    v1=np.concatenate((a,b,b,b,a,b,b,b),axis=0)
    v2=np.concatenate((b,b,a,b,b,b,a,b),axis=0)
    c1=np.float(np.dot(pos.reshape(1,-1),v1))
    c2=np.float(np.dot(pos.reshape(1,-1),v2))
    cost2=0.7*c1+0.3*c2

    pos=torch.Tensor(pos)

    cost=np.float(cost)
    #cost=2.0*cost+cost2*0.5
    cost=2.0*cost
    reward=-cost
    reward=np.array(reward).astype(np.float32)
    reward=torch.from_numpy(reward).unsqueeze(0)
    light=np.array(light).astype(np.float32)

    #############################################################################
    replay_memory = []
    #replay_memory= torch.load('GAT+RL/pretrained_model/replay_memory1000.pth')

    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1

 
    iteration = 0
    epsilon_decrements = np.linspace(model.initial_epsilon, model.final_epsilon, model.number_of_iterations)
    epsilon = model.initial_epsilon
 

    while iteration < model.number_of_iterations:

        if((light==[1,0]).all()):
            N9FirstHalf=np.zeros([1,24])
        else:
            N9FirstHalf=np.ones([1,24])
        longOflight=np.int(time.time()-time0)
        N9SecondHalf=np.ones([1,24])*longOflight/80.0
        Node9=np.concatenate((N9FirstHalf,N9SecondHalf),axis=1)
        Node9=Node9.astype(np.float32)
        Node9=torch.from_numpy(Node9)
        state=torch.cat((pos,Node9),0).to(device)
        
        output = model.forward(state,edge)
        
        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        if torch.cuda.is_available():
            action = action.cuda()

        random_action = random.random() <= epsilon
        if random_action:
            print("Performed random action!")
        action_index = [torch.randint(model.number_of_actions, torch.Size([]), dtype=torch.int)
                        if random_action
                        else torch.argmax(output.squeeze(0))][0]
        action[action_index] = 1

        if(action[0]==1):
            portx="COM7"
            bps=19200
            timex=5
            ser=serial.Serial(portx,bps,timeout=timex)
            ser.parity='E'
            begin=time.time()
            while ((time.time()-begin)<0.5):
                result=ser.write('1\0'.encode('gbk'))
            ser.close()   
        #############################################################################
        time.sleep(0.2)
        
        while(True):
            for k in [1,2,3,4,5,6,7,8]:
                fL=open('D:\DataRaw\DataSet\log_direction'+str(k)+'.txt','r')
                while(True):
                    temp=fL.readlines()
                    if np.size(temp)!=0: break
                fL.close()
                l = len(temp)
                for i in range(l): 
                    temp[i] = temp[i].strip() 
                    temp[i] = temp[i].strip('[]') 
                    temp[i] = temp[i].split(",")
                globals()['pos'+str(k)]=np.array(temp).astype(np.float32).reshape(1,48)
                if(k==1):pos_l=pos1
                else:   
                    pos_l=np.concatenate((pos_l,globals()['pos'+str(k)]),axis=0)
            if pos_l.shape==(8,48):
                fL.close()
                break    
        
        ##############################################################
        c1=np.float(np.dot(pos_l.reshape(1,-1),v1))
        c2=np.float(np.dot(pos_l.reshape(1,-1),v2))
        ##############################################################

        cost2=0.7*c1+0.3*c2
        pos_l=torch.Tensor(pos_l)
        
        fh = open('D:\cost.txt', 'r') 
        cost=fh.read()
        while(cost==''):
            cost=fh.read()
        fh.close()
        
        cost=np.float(cost)
        #cost=2.0*cost+cost2*0.5
        cost=2.0*cost
        if( action_index.cpu().detach().numpy()==0):
            cost=cost+10
        C=cost
        reward=-cost
       

        """New additional cost
        """
        if((light==[0,1]).all()):
            if(action[0]==1):
                if(c1>6):
                    reward=reward-100
        if((light==[1,0]).all()):
            if(action[1]==1):
                if((c2<6)&(longOflight>30)):
                    reward=reward-100       
        
        reward=np.array(reward).astype(np.float32)
        reward=torch.from_numpy(reward).unsqueeze(0)
                
        fL=open('D:\TrafficLight.txt','r')
        temp=fL.read()
        while(len(temp)<2):
            temp=fL.read()
        light_l=np.zeros(2)
        light_l[0]=temp[0]
        light_l[1]=temp[1]
        fL.close()
        if((sign!=light_l).all()):
            time0=time.time()
            sign=light_l
        light_l=np.array(light_l).astype(np.float32)
       
        #############################################################################
        action = action.unsqueeze(0)

        if((light_l==[1,0]).all()):
            N9FirstHalf=np.zeros([1,24])
        else:
            N9FirstHalf=np.ones([1,24])
        longOflight=np.int(time.time()-time0)
        N9SecondHalf=np.ones([1,24])*longOflight/80.0
        Node9=np.concatenate((N9FirstHalf,N9SecondHalf),axis=1)
        Node9=Node9.astype(np.float32)
        Node9=torch.from_numpy(Node9)
        state_l=torch.cat((pos_l,Node9),0)      

        replay_memory.append((state,action, reward, state_l))
            
        if len(replay_memory) > model.replay_memory_size:
            replay_memory.pop(0)

        epsilon = epsilon_decrements[iteration]

        minibatch = random.sample(replay_memory, min(len(replay_memory), model.minibatch_size))

        state_batch = torch.cat(tuple(d[0].unsqueeze(0) for d in minibatch)).to(device)
        action_batch= torch.cat(tuple(d[1] for d in minibatch)).to(device)
        reward_batch = torch.cat(tuple(d[2] for d in minibatch)).to(device)
        state_l_batch = torch.cat(tuple(d[3].unsqueeze(0) for d in minibatch)).to(device)
        
        output_l_batch = torch.cat(tuple(model(state_l_batch[d],edge).unsqueeze(0) for d in range(0,min(len(replay_memory), model.minibatch_size))),dim=0)

        y_batch = torch.cat(tuple(reward_batch[i].unsqueeze(0) + model.gamma * torch.max(output_l_batch[i].unsqueeze(0))
                                  for i in range(len(minibatch))))

        q_value = torch.sum(torch.cat(tuple(model(state_batch[d],edge).unsqueeze(0) for d in range(0,min(len(replay_memory), model.minibatch_size)))) * action_batch, dim=1)
        ########################################################################################
        globals()['ax'].append(iteration)
        globals()['ay'].append(-reward)
        plt.clf()
        plt.plot(ax,ay)
        plt.pause(0.1)
        plt.ioff()
        ########################################################################################
        optimizer.zero_grad()
        y_batch = y_batch.detach()
        loss = criterion(q_value, y_batch)        
        loss.backward()
        optimizer.step()

        state = state_l
        light=light_l
        iteration += 1
        
        if iteration % 1000 == 0:
            torch.save(replay_memory, "GAT+RL/pretrained_model/replay_memory" + str(iteration) + ".pth")
            torch.save(model, "GAT+RL/pretrained_model/current_model_" + str(iteration) + ".pth")
            np.savetxt('iteration'+str(iteration)+'.txt',globals()['ax'],fmt='%0.8f')
            np.savetxt('cost'+str(iteration)+'.txt',globals()['ay'],fmt='%0.8f')           
            plt.savefig(str(iteration)+".png")
        print("iteration:", iteration, "light time:", time.time() - time0,  "action:",
              action_index.cpu().detach().numpy(), "COST:",-reward.numpy() )

def test(model):

    fL=open('D:\TrafficLight.txt','r')
    temp=fL.read()
    light=np.zeros(2)
    light[0]=temp[0]
    light[1]=temp[1]
    fL.close()
    
    sign=light
    print('syncronizing light period')
    while((sign==light).all()):
        fL=open('D:\TrafficLight.txt','r')
        temp=fL.read()
        while(temp==''):
            temp=fL.read()
        light=np.zeros(2)
        light[0]=temp[0]
        light[1]=temp[1]
    sign=light
    time0=time.time()
    fL.close()

    ############################################
    '''
    path = "D:\log.csv"
    with open(path,'w',newline='') as f:
        csv_write = csv.writer(f)
        csv_head = ["time","n1","n2","action","N_pass","cost","time of video"]
        csv_write.writerow(csv_head)
    '''
    ############################################
    T_video=time.time()
    iteration=0
    while True:
        fh = open('D:\cost.txt', 'r') 
        cost=fh.read()
        while(cost==''):
            cost=fh.read()
        fh.close()

        fL=open('D:\TrafficLight.txt','r')
        temp=fL.read()
        while(len(temp)<2):
            temp=fL.read()
        light=np.zeros(2)
        light[0]=temp[0]
        light[1]=temp[1]
        fL.close()
        if((sign!=light).all()):
            time0=time.time()
            sign=light
        light=np.array(light).astype(np.float32)


    
        cost=np.float(cost)
        reward=-cost
        reward=np.array(reward).astype(np.float32)
        reward=torch.from_numpy(reward).unsqueeze(0)

        while(True):
            for k in [1,2,3,4,5,6,7,8]:
                fL=open('D:\DataRaw\DataSet\log_direction'+str(k)+'.txt','r')
                while(True):
                    temp=fL.readlines()
                    if (np.size(temp)!=0): break
                fL.close()
                l = len(temp)
                for i in range(l): 
                    temp[i] = temp[i].strip() 
                    temp[i] = temp[i].strip('[]') 
                    temp[i] = temp[i].split(",")
                globals()['pos'+str(k)]=np.array(temp).astype(np.float32).reshape(1,48)
                if(k==1):pos=pos1
                else:   
                    pos=np.concatenate((pos,globals()['pos'+str(k)]),axis=0)
            if pos.shape==(8,48):
                fL.close()
                break    

        pos=torch.Tensor(pos)

        if((light==[1,0]).all()):
            N9FirstHalf=np.zeros([1,24])
        else:
            N9FirstHalf=np.ones([1,24])
        longOflight=np.int(time.time()-time0)
        N9SecondHalf=np.ones([1,24])*longOflight/80.0
        Node9=np.concatenate((N9FirstHalf,N9SecondHalf),axis=1)
        Node9=Node9.astype(np.float32)
        Node9=torch.from_numpy(Node9)
        state=torch.cat((pos,Node9),0).unsqueeze(0).to(device)

        AdjacentMatrix=torch.Tensor(np.array([[1,0,0,0,0,0,0,0,1],[0,1,0,0,0,0,0,0,1],[0,0,1,0,0,0,0,0,1],[0,0,0,1,0,0,0,0,1],[0,0,0,0,1,0,0,0,1],[0,0,0,0,0,1,0,0,1],[0,0,0,0,0,0,1,0,1],[0,0,0,0,0,0,0,1,1],[1,1,1,1,1,1,1,1,1]]))
        edge=AdjacentMatrix.unsqueeze(0).to(device)   
        num_node = torch.ones(1).to(device) * 9        
        output = model.forward(state,edge, num_node)

        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        if torch.cuda.is_available():
            action = action.cuda()

        action_index = torch.argmax(output.squeeze(0))
        if torch.cuda.is_available():
            action_index = action_index.cuda()
        action[action_index] = 1
        '''
        if(action[0]==1):
            portx="COM7"
            bps=19200
            timex=5
            ser=serial.Serial(portx,bps,timeout=timex)
            ser.parity='E'
            begin=time.time()
            while ((time.time()-begin)<0.5):
                result=ser.write('1\0'.encode('gbk'))
            ser.close()
        '''
        print("iteration:", iteration,"light time:", time.time() - time0, "action:",action_index.cpu().detach().numpy())
        time.sleep(2)
        globals()['ax'].append(iteration)
        globals()['ay'].append(cost)
        plt.clf()
        plt.plot(ax,ay)
        plt.pause(0.1)
        plt.ioff()

        ##############################################
        '''
        with open(path,'a+',newline='') as f:
            csv_write = csv.writer(f)
            data_row = [np.str(np.int(time.time() - time0)),np.str(c1),np.str(c2),np.str(action_index.cpu().detach().numpy()),np.str(Npass),np.str(cost),np.str(time.time()-T_video)]
            csv_write.writerow(data_row)
            f.close()
        '''
        ##############################################



        iteration+=1

ax = []                    
ay = []

def main(mode):
    if mode == 'test':
        if torch.cuda.is_available():
            model = torch.load('GAT+RL/pretrained_model/current_model_6000.pth').eval()
        else:
            model = torch.load('GAT+RL/pretrained_model/current_model_6000.pth', map_location='cpu').eval()
        if torch.cuda.is_available():
            model = model.cuda()
        test(model)
    elif mode == 'train':
        if not os.path.exists('GAT+RL/pretrained_model/'):
            os.mkdir('GAT+RL/pretrained_model/')
       
        model = GAT(nfeat=48, nhid=4, nclass=2, dropout=0.6, alpha=0.2, nheads=1)
        model = model.to(device)

        '''
        if torch.cuda.is_available():
            model = torch.load('GAT+RL/pretrained_model/current_model_1000.pth').eval()
        else:
            model = torch.load('GAT+RL/pretrained_model/current_model_1000.pth', map_location='cpu').eval()
        if torch.cuda.is_available():
            model = model.cuda()
        '''
        start = time.time()
        train(model, start)


if __name__ == "__main__":
    #main(sys.argv[1])
    main('train')



