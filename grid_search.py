import sys, os, shutil, pdb, random, pickle
from tqdm import tqdm
def q(text = ''):
    print(f'>{text}<')
    sys.exit()

from environment import TicTacToe
from agent import QLearningAgent 
import config as cfg
from config import display_board
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt

hyperparameters_dict = {
                        'epsilon': np.arange(0.1, 0.4, 0.1),
                        'alpha'  : np.arange(0.1, 0.4, 0.1),
                        }

average_rewards = {
                   'player1': {},
                   'player2': {},
                  }

print('Grid-Searching for the following hyperparamets:')
pprint(hyperparameters_dict)

runs = 30
for epsilon in hyperparameters_dict['epsilon']:
    for alpha in hyperparameters_dict['alpha']:
        
        print(epsilon, alpha)
        for r in range(runs):

            env = TicTacToe()
            player1 = QLearningAgent(name = cfg.playerX_QLearningAgent_name, epsilon = epsilon, alpha = alpha)
            player2 = QLearningAgent(name = cfg.player0_QLearningAgent_name, epsilon = epsilon, alpha = alpha)

            episodes = cfg.num_episodes
            reward_player1_all_runs_and_episodes = np.zeros((runs, episodes))
            reward_player2_all_runs_and_episodes = np.zeros((runs, episodes))

            for i in range(episodes):
            # for i in tqdm(range(episodes)):
                
                if cfg.display:
                    print(f'TRAINING {str(i+1).zfill(len(str(episodes)))}/{episodes}')

                episode_reward_player1 = 0
                episode_reward_player2 = 0
                
                env.reset()
                player1.reset()
                done = False   
                playerID = random.choice([True, False]) # True means player1
                while not done:
                    # select player action
                    if playerID:
                        action = player1.epsilon_greedy(env.board, env.possible_actions())
                    else:
                        action = player2.epsilon_greedy(env.board, env.possible_actions())
                        # action = player2.choose_action(env.possible_actions())
                        
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
                            player1.updateQ(reward   , env.board, env.possible_actions())                
                            player2.updateQ(-1*reward, env.board, env.possible_actions())
                        else:
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
                    
                    elif reward == 0: # BUT WHY ?????????????????????????????? 
                        if not playerID:
                            player1.updateQ(reward, env.board, env.possible_actions())
                        else:
                            player2.updateQ(reward, env.board, env.possible_actions())

                    playerID = not playerID # switch turns

                # Logging the episode reward for both the players for all the runs (r) and episodes (i)
                reward_player1_all_runs_and_episodes[r, i] = episode_reward_player1
                reward_player2_all_runs_and_episodes[r, i] = episode_reward_player2
        
        # Logging average rewards for both the players for all possible sets of epsilons and alphas
        average_rewards['player1'][(epsilon, alpha)] = reward_player1_all_runs_and_episodes.mean(axis = 0)
        average_rewards['player2'][(epsilon, alpha)] = reward_player2_all_runs_and_episodes.mean(axis = 0)        
        
# # Plot the avg rewards for player1
# fig = plt.figure(constrained_layout=True)
# for epsilon_alpha_tuple in average_rewards['player1'].keys():
#     plt.plot(average_rewards['player1'][epsilon_alpha_tuple], label = f'epsilon {round(epsilon_alpha_tuple[0], 1)}, alpha {round(epsilon_alpha_tuple[1], 1)}')
# plt.xlabel('#episodes')
# plt.ylabel('avg rewards')
# plt.legend(loc = 'lower right')
# fig.savefig('averaged_rewards.png')
# plt.show()

with open('average_rewards.pickle', 'wb') as handle:
    pickle.dump(average_rewards, handle, protocol=pickle.HIGHEST_PROTOCOL)
print(f'\naverage_rewards saved !\n')