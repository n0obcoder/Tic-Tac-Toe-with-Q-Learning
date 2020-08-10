# Tic-Tac-Toe-with-Q-Learning
Training an agent to learn play Tic-Tac-Toe using Q-Learning

<p align="center">
<img src='/images/terminal_states_250x250.gif' width='350' alt='Losing is just not an option for the Tic-Tac-Toe master'>
</p>

## Requirements
* torch     -'1.5.0+cu101'         
* numpy     -1.19.0
* matplotlib-3.1.1  
* tqdm 

## How to Train the QLearning Agents
```
python main.py
```

## How to run Tensorboard for Visualization
```
tensorboard --logdir <summary_directory_path>
```
or
```
tensorboard --logdir summary
```
because we <b>"summary"</b> set as the summary_directory_path in <b>config.py</b>

## How to Play TicTocToe Against a QLearning Agent
```
python play.py
```

## Result

Reward plots for both the QLearningAgents over the course of 100,000 episodes.

<p align="center">
<img src='/images/rewards.PNG'  alt='Losing is just not an option for the Tic-Tac-Toe master'>
</p>
