import airsim
import cv2
import numpy as np
import os
import time
import serial
import inspect
import math

client = airsim.CarClient()
car_controls = airsim.CarControls()

road8=[[-17,-4],[-63,-4],-180]
road1=[[-63,4],[-17,4],0]

road2=[[-4,17],[-4,63],90]
road3=[[4,63],[4,17],-90]

road4=[[17,4],[63,4],0]
road5=[[63,-4],[17,-4],-180]

road6=[[4,-17],[4,-63],-90]
road7=[[-4,-63],[-4,-17],90]

road9=[[0,0],[0,0],0]#Undefined

intersection=[[-17,-17],[17,17]]

def ToEulerAngles(w,x,y,z):
    y = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    y = y / math.pi * 180
    return y

def get_distance_from_point_to_line(point, line_point1, line_point2):
    A = line_point2[1] - line_point1[1]
    B = line_point1[0] - line_point2[0]
    C = (line_point1[1] - line_point2[1]) * line_point1[0] + (line_point2[0] - line_point1[0]) * line_point1[1]
    distance = (A * point[0] + B * point[1] + C) / (np.sqrt(A**2 + B**2)+1e-6)
    return distance

def InRegion(p,point1,point2):
 
    if(p[0]<=np.min([point1[0],point2[0]])-0.1):
        	return False;
    if(p[0]>=np.max([point1[0],point2[0]])+0.1):
        	return False;
    if(p[1]>=np.max([point1[1],point2[1]])+0.1):
        	return False;
    if(p[1]<=np.min([point1[1],point2[1]])-0.1):
        	return False;
    return True;

def InIntersection(p):
 
    if(p[0]<=np.min([globals()['intersection'][0][0],globals()['intersection'][1][0]])):
        	return False;
    if(p[0]>=np.max([globals()['intersection'][0][0],globals()['intersection'][1][0]])):
        	return False;
    if(p[1]>=np.max([globals()['intersection'][0][1],globals()['intersection'][1][1]])):
        	return False;
    if(p[1]<=np.min([globals()['intersection'][0][1],globals()['intersection'][1][1]])):
        	return False;
    return True;

PNC1=1
PNC2=1
#position of car
NextSection1=0
NextSection2=0
client.enableApiControl(True, "Car1")
#client.enableApiControl(True, "Car2")

stack_OnRoad=[1]
while (True):

    for i in stack_OnRoad:
        """This part cover's realizations of cars' basic moving logics.
        It includes:
            1. Making cars driving along roads
            2. Cars maybe turns left under a certain prob. at intersections. 

        Only care about steering.
        """
        car_state = client.getCarState('Car'+str(i))
    
        if (car_state.speed < 7):
            car_controls.throttle = 1.0
        else:
            car_controls.throttle = 0.0

        pose=client.simGetObjectPose('Car'+str(i))
        pos=[pose.position.x_val,pose.position.y_val]    
    
        if(InIntersection(pos)&(globals()['NextSection'+str(i)]==0)):
            if(InRegion(pos,globals()['target'+str(i)][0],globals()['target'+str(i)][1])):
                dist=get_distance_from_point_to_line(pos,globals()['target'+str(i)][0],globals()['target'+str(i)][1])
                DegreesOfSection=math.degrees(math.atan(np.abs((globals()['target'+str(i)][1][1]-globals()['target'+str(i)][0][1])/(globals()['target'+str(i)][1][0]-globals()['target'+str(i)][0][0]))))
                CarOri=ToEulerAngles(pose.orientation.w_val,pose.orientation.x_val,pose.orientation.y_val,pose.orientation.z_val)
                if((globals()['PNC'+str(i)]==1)):
                    DegreesOfSection=DegreesOfSection            
                if((globals()['PNC'+str(i)]==3)):
                    DegreesOfSection=-90+DegreesOfSection
                if((globals()['PNC'+str(i)]==5)):
                    DegreesOfSection=-180+DegreesOfSection
                    if(CarOri>=0):
                        CarOri=(CarOri-360)
                if((globals()['PNC'+str(i)]==7)):
                    DegreesOfSection=180-DegreesOfSection
                car_controls.steering=np.tanh(dist)-0.2*(CarOri-DegreesOfSection)
                if (car_state.speed > 1.8):
                    car_controls.throttle = 0.0
                approve=True
            else:
                if(approve):
                    globals()['NextSection'+str(i)]=1################
        else:
            globals()['PNC'+str(i)]+=globals()['NextSection'+str(i)]##############
            globals()['NextSection'+str(i)]=0
            approve=False

            dist=get_distance_from_point_to_line(pos,globals()['road'+str(globals()['PNC'+str(i)])][0],globals()['road'+str(globals()['PNC'+str(i)])][1])
            CarOri=ToEulerAngles(pose.orientation.w_val,pose.orientation.x_val,pose.orientation.y_val,pose.orientation.z_val)
        
        
            if((globals()['PNC'+str(i)]==5)|(globals()['PNC'+str(i)]==8)):
                if(CarOri>=0):
                    CarOri=(CarOri-360)
            CompensateAngle=0.2*(CarOri-globals()['road'+str(globals()['PNC'+str(i)])][2])    
            car_controls.steering=5*np.tanh(dist)-CompensateAngle           

            globals()['target'+str(i)]=[globals()['road'+str(globals()['PNC'+str(i)])][1],globals()['road'+str(globals()['PNC'+str(i)]+1)][0]]########
            #target marks the begining and the ending points of a road section 

            """
            This part covers realizations of cars' control logics

            """
            
        for j in stack_OnRoad:
            if(j!=i):
                if(globals()['PNC'+str(j)]==globals()['PNC'+str(i)]):
                    pose=client.simGetObjectPose('Car'+str(j))
                    pos_next=[pose.position.x_val,pose.position.y_val]
                    distance=np.sqrt(np.square(pos[0]-pos_next[0])+np.square(pos[1]-pos_next[1]))
                    if(distance<=8):
                        car_controls.brake=1
                        car_controls.throttle = 0
                    else:
                        car_controls.brake=0
            
        #pass the intersection:using a vector to record which roads can pass accroding to tafficlight.
        #Car i will stop when there is a car of horizontal directions in the intersection unless it's PNC in the vector.





        
        client.setCarControls(car_controls,'Car'+str(i))
            