import sys, os, shutil, pdb, random
from tqdm import tqdm
def q(text = ''):
    print(f'>{text}<')
    sys.exit()

from environment import TicTacToe
from agent import QLearningAgent, Hoooman
import config as cfg
from config import display_board

# initializing the TicTacToe environment and a QLearningAgent (the master Tic-Toc-Toc player, your opponent!)
env = TicTacToe()
player1 = QLearningAgent(name = cfg.playerX_QLearningAgent_name)
player1.loadQtable() # load the learnt Q-Table
player1.epsilon = 0.0 # greedy actions only, 0 exploration

# initializing the agent class that let's you, the human user take the actions in the game 
player2 = Hoooman() 

# replay decides whether to rematch or not, at the end of a game 
replay = True
while replay:

    done = False   # the episode goes on as long as done is False
    
    # deciding which player makes a move first    
    playerID = random.choice([True, False]) # True means player1
    
    while not done:
        # select player action by using epsilon-greedy algorithm, depending on the environment's board configuration and the possible actions available to the player
        if playerID:
            action = player1.epsilon_greedy(env.board, env.possible_actions())
        else:
            # human user takes an action my entering one of the possible inputs in the terminal 
            print(f'\nPossible Actions: {env.possible_actions()}')
            action = int(input('Select an action ! '))
            
        # take selected action
        reward, done = env.step(playerID, action)

        # display board
        display_board(env.board, action, playerID, player1, player2, reward, done, env.possible_actions(), training=False)

        playerID = not playerID # switch turns

    # asks the human user if she/he wants to play another match
    replay = input('\nPlay Again ? [y/n] ')
    if replay.lower() == 'y':
        # resetting the environemnt and the Q-Learning agent if the human user choses to play another match 
        env.reset()     
        player1.reset() 
        print('\n-----------------------------NEW GAME-----------------------------')

    elif replay.lower() == 'n':
        # setting replay to False if the human user choses not to play another match
        replay = False  
        print('Thank you for wasting your time :)') 