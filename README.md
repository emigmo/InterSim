# InterSim
## Prerequisites and setup
### Environment Setup
1. Install [Unreal Engine 4.23.1](https://www.unrealengine.com/). Make sure the version of editor is 4.23.1.

2. Install Microsoft AirSim as a plugin in UE4. You can follow the  instruction on AirSim handbook
	```bash
	https://microsoft.github.io/AirSim/
	```
3. Download scene of an urban crossroads system  that we made for this experiment.
	`
	.\AD_Cookbook_Start_AirSim.ps1 landscape
	`

### How to run the simulator

1. Replace setting.json of AirSim with the file we provided.

2. Run the scene in UE4.

3. Run CarControls.py and you will see the car flow on map.

4. You can choose either Training_dqn.py or Testing_dqn.py to  reproduce our expeirments.


## Build the Simulator of Traffic Intersection
0. *Airsim & UE4*
```
todo
```
1. *Traffic Map Build*
```
import ue4 file (file link)
```
2. *Car Control*
```
todo
```
3. *Traffic Light Control*
```
todo
```

## Reinforcement Learning for Intelligence Traffic Light Control

1. *Single Intersection*
```
DQN or other RL Approach
```
