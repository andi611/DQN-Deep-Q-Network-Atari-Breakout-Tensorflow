# Reinforcement Learning: Deeq Q Learning Network (DQN) Agent playing Atari Breakout
* Training a vision-based agent with the Deep Q Learning Network (DQN) in Atari's Breakout environment, implementation in Tensorflow.
![DQN Agent Game Play Demo](https://github.com/andi611/Reinforcement-Learning-DQN-Deep-Q-Learning-Atari-Breakout/blob/master/model/gameplay.gif)
<img src="https://github.com/andi611/Reinforcement-Learning-DQN-Deep-Q-Learning-Atari-Breakout/blob/master/model/gameplay.gif" width="329" height="478">
## Environment
* **< Python 3.7 >**
* **< [OpenAI Gym](https://github.com/openai/gym) >**
	- Install the OpenAI Gym Atari environment:
	`$ pip3 install opencv-python gym "gym[atari]"`
	- Atari environment used: `BreakoutNoFrameskip-v4`
* **< [Tensorflow r.1.12.0](https://www.tensorflow.org/) >**

## Implementation
* Deep Q Learning Network with the following improvements:
	- Experience Replay
	- Fixed Target Q-Network
	- TD error loss function with: *Q<sub>target</sub> = reward + (1-terminal) * (gamma * Q<sub>max</sub>(s’) )*
* DQN network Settings:
![](https://github.com/andi611/Reinforcement-Learning-DQN-Deep-Q-Learning-Atari-Breakout/blob/master/model/dqn_best_setting.png)

## File Description
```
.
├── ./
|   ├── agent_dir.py ─────────── DQN model
|   ├── atari_wrapper.py ─────── Atari wrapper
|   ├── environment.py ───────── Gym wrapper
|   ├── runner.py ────────────── Main program for training and testing
|   └── Readme.md ────────────── This file
└── model/
	├── dqn_learning_curve_compare.png ──────── Figure 1  
	├── dqn_best_setting.png ────────────────── Figure 2
	├── dqn_learning_curve.png ──────────────── Figure 3
	├── checkpoint ──────────────────────────── Tensorflow model check point
	├── model_dqn-25581.data-00000-of-00001 ─── Tensorflow model data
	├── model_dqn-25581.meta ────────────────── Tensorflow model meta
	└── model_dqn-25581.index ───────────────── Tensorflow model index
```

## Usage
* Traing the DQN Agent: `$ python3 runner.py --train_dqn`
* Testing the DQN Agent: `$ python3 runner.py --test_dqn`
* Testing the DQN Agent with **gameplay rendering**: `$ python3 runner.py --test_dqn --do_render`

## Learning Curve
* Single learning curve:
![](https://github.com/andi611/Reinforcement-Learning-DQN-Deep-Q-Learning-Atari-Breakout/blob/master/model/dqn_learning_curve.png)
* With different plotting window:
![](https://github.com/andi611/Reinforcement-Learning-DQN-Deep-Q-Learning-Atari-Breakout/blob/master/model/dqn_learning_curve_compare.png)