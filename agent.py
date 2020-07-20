import random, pickle
import config as cfg

class Hoooman():
    def __init__(self):
        self.name = 'Hoooman'

class RandomActionAgent():
    def __init__(self, name = 'RandomActionAgent'):
        self.name = name

    def choose_action(self, possible_actions):
        # random action selection
        action = random.choice(possible_actions)
        return action

class QLearningAgent:
    def __init__(self, name, epsilon = cfg.epsilon, alpha = cfg.alpha, gamma = cfg.gamma):
        self.name = name
        self.epsilon = epsilon
        self.alpha   = alpha
        self.gamma   = gamma
        self.Q       = {} # Q-Table
        self.last_board = None
        self.state_action_last = None
        self.q_last = 0.0

    def reset(self):
        '''
        resets the last_board, state_action_last and q_last 
        '''
        self.last_board = None
        self.state_action_last = None
        self.q_last = 0.0

    def epsilon_greedy(self, state, possible_actions):
        '''
        returns action by using epsilon-greedy algorithm
        '''
        # print('in epsilon-greedy, possible_actions: ', possible_actions)

        state = tuple(state) # because state is going to be a part of the key of Q-dict and dictionaries can not have lists as the keys
        self.last_board = state # 
        if random.random() < self.epsilon:
            # random action selection
            action = random.choice(possible_actions)
            
        else:
            # greedy action selection
            q_list = [] 
            # we will store q-values for all the possiblle actions available for the current state
            for action in possible_actions:
                q_list.append(self.getQ(state, action))
            maxQ = max(q_list)

            # print('q_list: ', q_list)

            # we need to handle the cases where more than 1 action has the same maxQ
            if q_list.count(maxQ) > 1:
                # in case when we have more than 1 action having the same maxQ, we randomly pick one of those actions
                maxQ_actions = [i for i in range(len(possible_actions)) if q_list[i] == maxQ]
                action_idx = random.choice(maxQ_actions)
            else:
                # in case when we have only 1 action having the same maxQ, simply pick that action
                action_idx = q_list.index(maxQ)

            action = possible_actions[action_idx]

        # update state_action_last and q_last        
        self.state_action_last = (state, action) 
        self.q_last = self.getQ(state, action)

        return action

    def getQ(self, state, action):
        '''
        return q-value for a given state-action pair
        '''
        return self.Q.get((state, action), 1.0) 

    def updateQ(self, reward, state, possible_actions):
        '''
        performs Q-Learning update
        '''
        q_list = []
        for action in possible_actions:
            q_list.append(self.getQ(tuple(state), action))
        
        if q_list:
            maxQ_next = max(q_list)
        else:
            maxQ_next = 0

        # Q-Table update
        self.Q[self.state_action_last] = self.q_last + self.alpha*((reward + self.gamma*maxQ_next) - self.q_last)

    def saveQtable(self):
        '''
        saves the Q-Table as a pickle file
        '''
        save_name = self.name + '_QTable'
        with open(save_name, 'wb') as handle:
            pickle.dump(self.Q, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'\nQ-Table for {self.name} saved as >{save_name}<\n')

    def loadQtable(self): # load table
        '''
        loads the Q-Table from a pickle file
        '''
        load_name = self.name + '_QTable'
        with open(load_name, 'rb') as handle:
            self.Q = pickle.load(handle)
        print(f'\nQ-Table for {self.name} loaded as >{load_name}< B)\n')