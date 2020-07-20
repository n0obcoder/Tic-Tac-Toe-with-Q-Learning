from agent import QLearningAgent

class TicTacToe:
    def __init__(self):
        '''
        the environment starts with 9 empty spaces representing a board of Tic-Tac-Toe 
        '''
        self.board = ['_']*9   # the initial blank board
        self.done  = False     # done = True means the game has ended        

    def reset(self):
        '''
        resets the board for a new game
        '''
        self.board = ['_']*9

    def possible_actions(self):
        '''
        returns a list of possible actions that can be taken on the current board
        '''
        return [moves + 1 for moves, v in enumerate(self.board) if v == '_']

    def step(self, playerID, action):
        '''
        takes the action for the player with the given playerID resulting in a change in the board configuration.  
        it finally returns the reward and done status by evaluating the board congiguration. 
        '''
        if playerID:
            ch = 'X'
        else:
            ch = '0'

        if self.board[action - 1] != '_': # negative reward for picking the action which corresponds to marking the square which is already marked
            return -5, True

        self.board[action - 1] = ch

        # reward, done = self.evaluate(playerID)
        return self.evaluate(playerID)

    def evaluate(self, playerID):
        '''
        returns reward and done status for the player with the given playerID by evaluating the board congiguration.
        '''  
        if playerID:
            ch = 'X'
        else:
            ch = '0'

        # WIN CONDITIONS
        # rows checking        
        for i in range(3):
            if (ch == self.board[(i*3)+0] == self.board[(i*3)+1] == self.board[(i*3)+2]):
                # print(f'---row num: {i}')
                return 1, True
        
        # cols checking        
        for i in range(3):
            if (ch == self.board[i+0] == self.board[i+3] == self.board[i+6]):
                # print(f'---col num: {i}')
                return 1, True

        # diagonal checking
        if (ch == self.board[0] == self.board[4] == self.board[8]):
            # print('---diagonal 1')
            return 1.0, True

        if (ch == self.board[2] == self.board[4] == self.board[6]):
            # print('---diagonal 2')
            return 1.0, True

        # DRAW CONDITION
        # if all positions are filled
        if not any(c == '_' for c in self.board):
            # print('---all positions filled')
            return 0.5, True
        
        # GAME-ON CONDITION
        return 0, False