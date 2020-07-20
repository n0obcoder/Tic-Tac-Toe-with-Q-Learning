summary_dir = 'summary'
num_episodes = 500000
display = False

epsilon = 0.4
alpha = 0.3
gamma = 0.95

playerX_QLearningAgent_name = 'QLearningAgent_X'
player0_QLearningAgent_name = 'QLearningAgent_0'

def display_board(board, action, playerID, player1, player2, reward, done, possible_actions, training = True, episode_reward_player1=None, episode_reward_player2=None):
    '''
    prints out the Tic-Tac-Toe board in the terminal.
    prints the action taken by the players, the reward they recieved and the status of the game (Done ->  True or False)
    prints if either of the players have won or lost the game or if it is a tied between the players
    prints all the possible next actions if the training argument is set to True
    '''
    print('\n')
    for i in range(3):
        print('  '.join(board[i*3:(i+1)*3]))
    
    player = player1.name if playerID else player2.name
    print(f'{player} takes action {action}, gets reward {reward}. Done -> {done}')
    if episode_reward_player1 is not None:
        print(f'episode_reward_player1 -> {episode_reward_player1}')
        print(f'episode_reward_player2 -> {episode_reward_player2}')
    if reward < 0:
        if playerID:
            print(f'\n{player2.name} wins !')
        else:
            print(f'\n{player1.name} wins !')
    elif reward == 1:
        if playerID:
            print(f'\n{player1.name} wins !')
        else:
            print(f'\n{player2.name} wins !')
    elif reward == 0.5:
        print(f'\nit is a draw !')
    else:
        if training:
            print(f'\npossible_actions: {possible_actions}\n')