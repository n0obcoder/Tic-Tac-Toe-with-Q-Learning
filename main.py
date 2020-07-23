import sys, os, shutil, pdb, random
from tqdm import tqdm

def q(text = ''):
    '''
    a function that exits the code after printing a message. used for dubugging purposes
    '''
    print(f'>{text}<') # f-strings work only with python3
    sys.exit()

from environment import TicTacToe
from agent import QLearningAgent, RandomActionAgent 
import config as cfg
from config import display_board

# it is a little tricky on run SummaryWriter by installing a suitable version of pytorch. so if you are able to import SummaryWriter from torch.utils.tensorboard, this script will record summaries. Otherwise it would not.
try:
    from torch.utils.tensorboard import SummaryWriter
    write_summary = True
except:
    print('--------------------------------------------------------')
    print('You do not have SummaryWriter in torch.utils.tensorboard')
    print('This code will run anyway. It is just that the summaries would not be recorded')
    print('For the summaries to get recorded, install a suitable version of pytorch which has SummaryWriter in torch.utils.tensorboard')
    print('I had torch: 1.5.0+cu101 installed on my machine and it worked fine for me')
    print('--------------------------------------------------------')
    write_summary = False

# cfg.summary_dir is the path of the directory where the tensorboard SummaryWriter files are written
# the directory is removed, if it already exists
if write_summary:
    if os.path.exists(cfg.summary_dir):
        shutil.rmtree(cfg.summary_dir)

    writer = SummaryWriter(cfg.summary_dir) # this command automatically creates the directory at cfg.summary_dir

# initializing the TicTacToe environment and 2 QLearningAgent
env = TicTacToe()
player1 = QLearningAgent(name = cfg.playerX_QLearningAgent_name)
player2 = QLearningAgent(name = cfg.player0_QLearningAgent_name)
# player2 = RandomActionAgent()

episodes = cfg.num_episodes
for i in tqdm(range(episodes)):
    
    if cfg.display:
        print(f'TRAINING {str(i+1).zfill(len(str(episodes)))}/{episodes}')

    # initializing the episode reward for both the players to 0 
    episode_reward_player1 = 0
    episode_reward_player2 = 0
    
    # resetting the environemnt and both the agents 
    env.reset()
    player1.reset()
    player2.reset()

    done = False # the episode goes on as long as done is False

    # deciding which player makes a move first
    playerID = random.choice([True, False]) # True means player1
    
    while not done:
        # select player action by using epsilon-greedy algorithm, depending on the environment's board configuration and the possible actions available to the player
        if playerID:
            action = player1.epsilon_greedy(env.board, env.possible_actions())
        else:
            action = player2.epsilon_greedy(env.board, env.possible_actions())
            # action = player2.choose_action(env.possible_actions()) # action selection for RandomActionAgent
            
        # take selected action
        reward, done = env.step(playerID, action)

        if playerID:
            episode_reward_player1 += reward
        else:
            episode_reward_player2 += reward

        # display board
        if cfg.display:
            display_board(env.board, action, playerID, player1, player2, reward, done, env.possible_actions(), episode_reward_player1, episode_reward_player2, training = True)

        # Q-Table update based on the reward
        if reward == 1: # WIN n LOSE
            if playerID:
                # player1 wins and player2 loses
                player1.updateQ(reward   , env.board, env.possible_actions())                
                player2.updateQ(-1*reward, env.board, env.possible_actions())
            else:
                # player2 wins and player1 loses                
                player1.updateQ(-1*reward, env.board, env.possible_actions())
                player2.updateQ(reward, env.board, env.possible_actions())

        elif reward == 0.5: # DRAW
            player1.updateQ(reward, env.board, env.possible_actions())                
            player2.updateQ(reward, env.board, env.possible_actions())                

        elif reward == -5: # ILLEGAL ACTION
            if playerID:    
                player1.updateQ(reward, env.board, env.possible_actions())                
            else:
                player2.updateQ(reward, env.board, env.possible_actions())
        
        elif reward == 0: 
            if not playerID:
                player1.updateQ(reward, env.board, env.possible_actions())
            else:
                player2.updateQ(reward, env.board, env.possible_actions())

        # switch turns
        playerID = not playerID 

    if write_summary:
        # write tensorboard summaries
        writer.add_scalar(f'episode_reward/{player1.name}', episode_reward_player1, i)
        writer.add_scalar(f'episode_reward/{player2.name}', episode_reward_player2, i)

# save Q-Tables for both the players. either of these could be used as an opponent by a user.
player1.saveQtable()
player2.saveQtable()
