from Neural_Net import Neural_Net
from Point import Point

class Player(object):
    
    def __init__(self, team):
        '''
        Creates a player given their team. Team 
        should be 1 for X or -1 for O. 
        '''
        abstract()
    
    def close(self):
        '''
        Ties up loose ends.
        '''
        abstract()
    
    def draw(self):
        '''
        Tell the player they were adequate.
        '''
    
    def make_move(self, board):
        '''
        Calculates the next move the player will take. 
        Returns the confidence level of the previous 
        move if the player is a neural net. And returns 
        weather the move was successful.
        '''
        abstract()
    
    def punish(self):
        '''
        Punish the player for losing the game.
        '''
        abstract()
    
    def reward(self):
        '''
        Rewards the player for winning the game.
        '''
        abstract()
    
    def set_team(self, new_team):
        '''
        Sets the players team to the new team.
        '''
        self.team = new_team
    
    def _ask(self, question):
        '''
        Prompts the user with the given question. 
        Returns the output as a string.
        '''
        read_in = input(question)
        print()
        return read_in
    
    def _team_letter(self):
        '''
        Returns the letter (X or O) which which the 
        current player is associated.
        '''
        if self.team == 1 :
            return 'X'
        elif self.team == -1 :
            return 'O'
        else:
            raise('Team indeterminate.')

class NN_Player(Player):
    
    def __init__(self, team, filename=None, type=None):
        '''
        Requests a neural net file name from the user, 
        assuming the file is in the Saved_Neural_Nets/TTT/ 
        subdirectory. Asks
        '''
        self.team = team
        
        if filename == None :
            subdir = 'Saved_Neural_Nets/TTT/'
            question = 'Which file from ' + subdir + ' should play on team ' + self._team_letter() + '?\n'
            self.filename = subdir + self._ask(question)
        else:
            self.filename = filename
        
        if type == None :
            question = 'Should it play its best [b] or be somewhat random [r]?\n'
            type = self._ask(question)
        if type == 'b' :
            self.type = 1
        elif type == 'r' :
            self.type = 2
        else:
            raise('Type indeterminate.')
            
        self.nn = Neural_Net(filename=self.filename)
        if self.nn.layers[0] != 9 or self.nn.layers[len(self.nn.layers)-1] != 9 :
            raise Exception("Given neural nets must start and end with 9 input nodes.")
    
    def close(self):
        if self.filename != None :
            self.nn.save(self.filename)
    
    def draw(self):
        self.nn.reward(-0.1)
    
    def make_move(self, board):
        input = board.as_vector(self.team)
        valid = board.get_valid_vector()
        output, conf = self.nn.train_valid(input, valid, self.type)
        return conf, self._apply_move(board, output)
    
    def punish(self):
        self.nn.reward(-1.)
    
    def reward(self):
        self.nn.reward(1.)
    
    def _apply_move(self, board, output_vector):
        '''
        Places the player's marker at the point based 
        on the output_vector for the neural net 
        whose turn it is. Only one value in the 
        output_vector should be 1. All others 
        should be zero. Returns True if placement was 
        successful. False otherwise.
        '''
        for i in range(len(output_vector)):
            if output_vector[i] == 1 :
                pos = i
                break;
        r = 0
        while pos >= 3 :
            r = r + 1
            pos = pos - 3
        return board.place(self.team, Point(r, pos))

class Human_Player(Player):
    
    def __init__(self, team):
        self.team = team
        i = 1
        output = ''
        for r in range(3) :
            output = output + '['
            for c in range(3):
                output = output + str(i)
                if c < 2 :
                    output = output + ' '
                i = i + 1
            output = output + ']\n'
        print('These are the indexes of the position on the board:')
        print(output)
        print()
    
    def close(self):
        print('Bye')
    
    def draw(self):
        p = self._team_letter() + ', that was OK...'
        print(p)
    
    def make_move(self, board):
        index = self._prompt(board) - 1
        return None, self._apply_move(board, index)
    
    def punish(self):
        p = 'That was bad ' + self._team_letter() + '.\nBut better luck next time!'
        print(p)
    
    def reward(self):
        p = self._team_letter() + ', that was awesome!'
        print(p)
    
    def _apply_move(self, board, index):
        '''
        Places the players marker on the board 
        at the index specified.
        '''
        r = 0
        while index >= 3 :
            r = r + 1
            index = index - 3
        return board.place(self.team, Point(r, index))
    
    def _prompt(self, board):
        '''
        Prompts the user with which index they would next like to make a move. Returns the result as a int.
        '''
        t = self._team_letter()
        index = eval(self._ask("Enter index to place an " + t + ":"))
        print()
        return index