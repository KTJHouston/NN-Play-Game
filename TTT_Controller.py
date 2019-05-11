from Neural_Net import Neural_Net
from Tic_Tac_Toe import Tic_Tac_Toe as TTT
from Player import Player, NN_Player, Human_Player
import os
import math
import random

class TTT_Controller(object):
    X = 1
    O = -1
    
    def __init__(self):
        self._prompt_cycle()
        
    def _ask(self, question):
        '''
        Prompts the user with the given question. 
        Returns the output as a string.
        '''
        read_in = input(question)
        print()
        return read_in
    
    def _make_move(self):
        '''
        Makes a move for the players whose turn it is.
        '''
        if self.board.get_turn() == self.p1.team :
            conf, is_valid = self.p1.make_move(self.board)
        else:
            conf, is_valid = self.p2.make_move(self.board)
        return conf, is_valid
        
    def _pp_confidence(self, flist):
        '''
        Prints a list of floats, but restricts the number 
        of decimal places to 2.
        '''
        print('[', end='')
        for f in range(len(flist)-1):
            print(f'%.2f, ' % flist[f], end='')
        print(f'%.2f]' % flist[len(flist)-1])
        
    def _prompt_cycle(self):
        '''
        Prompts the user with what kind of game they would 
        like to play, then plays that game. Continues prompting 
        the user until they specificy they would like to Quit [q]. 
        '''
        played_game = False
        X = self.X
        O = self.O
        while True:
            q = "What type of mode would you like to run?"
            if played_game:
                q = q = "\n Play again [pa]"
            q = q + "\n Computer v Computer [cc]"
            q = q + "\n Human v Computer [hc]"
            q = q + "\n Human v Human [hh]"
            q = q + "\n Training [t]"
            q = q + "\n Create new neural nets [c]"
            q = q + "\n Quit [q]\n"
            input = self._ask(q)
            if input == 'pa' and played_game :
                self._single_game(True)
                self.p1.close()
                self.p2.close()
            elif input == 'cc' :
                self.p1 = NN_Player(X)
                self.p2 = NN_Player(O)
                self._single_game(True)
                self.p1.close()
                self.p2.close()
                played_game = True
            elif input == 'hc' :
                question = 'Which team would you like to be on [X/O]?\n'
                answer = self._ask(question)
                if answer == 'X' or answer == 'x' :
                    self.p2 = NN_Player(O)
                    self.p1 = Human_Player(X)
                elif answer == 'O' or answer == 'o' :
                    self.p1 = NN_Player(X)
                    self.p2 = Human_Player(O)
                self._single_game(True)
                self.p1.close()
                self.p2.close()
                played_game = True
            elif input == 'hh' :
                self.p1 = Human_Player(X)
                self.p2 = Human_Player(O)
                self._single_game(True)
                self.p1.close()
                self.p2.close()
                played_game = True
            elif input == 't' :
                self._prompt_training()
                played_game = False
            elif input == 'c' :
                self._prompt_create()
                played_game = False
            elif input == 'q' :
                break
            else:
                print('User Input was not recognized.')
    
    def _prompt_create(self):
        '''
        Prompts the user with the type of neural net 
        creation they would like to complete.
        '''
        question = 'Would you like to create a single neural net [s] or a batch or neural nets [b]?\n'
        answer = self._ask(question)
        if answer == 's' :
            self._prompt_create_single()
        elif answer == 'b' :
            self._prompt_create_batch()
    
    def _prompt_create_batch(self):
        '''
        Prompts the user with which attributes to create a 
        batch of neural nets.
        '''
        subdir = 'Saved_Neural_Nets/TTT/'
        question = 'To which folder in ' + subdir + ' would you like to save the batch of neural nets?\n'
        folder = subdir + self._ask(question)
        try:
            if not os.path.exists(folder):
                os.makedirs(folder)
        except OSError:
            print('Error: Creating directory. ' + directory)
        
        question = 'Would you like to create common [c] or varying [v] neural nets?\n'
        answer = self._ask(question)
        if answer == 'c' :
            self._prompt_create_batch_common(folder)
        elif answer == 'v' :
            self._prompt_create_batch_variety(folder)
    
    def _prompt_create_batch_common(self, folder):
        question = 'How many hidden layers should these neural nets have? (Separate with commas.)\n'
        layer_str = self._ask(question)
        
        layers = [9]
        n = ''
        for ls in layer_str :
            if ls == ',':
                layers.append(eval(n))
                n = ''
            else:
                n = n + ls
        if n != '' :
            layers.append(eval(n))
        layers.append(9)
        
        question = 'What should be the learning rate of these neural nets?\n'
        learning_rate = eval(self._ask(question))
        
        question = 'How many neural nets should be created?\n'
        number = eval(self._ask(question))
        
        print('Creating...')
        for n in range(number):
            nn = Neural_Net(layers, learning_rate)
            nn.save(folder + str(n) + '.json')
    
    def _prompt_create_batch_variety(self, folder):
        '''
        Prompts the user with necessary information to 
        create neural nets from a file of layer information.
        '''
        subdir = 'Saved_Neural_Nets/TTT/'
        question = 'Where in ' + subdir + ' is there a file specifying the layers of ?\n'
        file = subdir + self._ask(question)
        #TODO finish this method
    
    def _prompt_create_single(self):
        '''
        Prompts the user with which attributes to create 
        a neural net.
        '''
        subdir = 'Saved_Neural_Nets/TTT/'
        question = 'Where in ' + subdir + ' would you like to save the neural net?\n'
        filename = subdir + self._ask(question)
        
        question = 'How many hidden layers should this neural net have? (Separate with commas.)\n'
        layer_str = self._ask(question)
        
        layers = [9]
        n = ''
        for ls in layer_str :
            if ls == ',':
                layers.append(eval(n))
                n = ''
            else:
                n = n + ls
        if n != '' :
            layers.append(eval(n))
        layers.append(9)
        
        question = 'What should be the learning rate of this neural net?\n'
        learning_rate = eval(self._ask(question))
        
        nn = Neural_Net(layers, learning_rate)
        nn.save(filename)
    
    def _prompt_training(self):
        '''
        Prompts the user with the type of training they 
        would like to run.
        '''
        question = 'Would you like to train a pair [p] or batch [b]?\n'
        answer = self._ask(question)
        if answer == 'p' :
            self._prompt_training_pair()
        elif answer == 'b' :
            self._prompt_training_batch()
        else:
            print('User Input was not recognized: ' + answer)
    
    def _prompt_training_batch(self):
        '''
        Prompts the user with the type of batch training 
        to perform.
        '''
        question = 'Would you like to do even [e] or random [r] matchmaking?\n'
        answer = self._ask(question)
        
        subdir = 'Saved_Neural_Nets/TTT/'
        question = 'In which folder in ' + subdir + ' is the batch of neural nets located?\n'
        folder = subdir + self._ask(question)
        
        if answer == 'e':
            self._prompt_training_batch_even(folder)
        elif answer == 'r':
            self._prompt_training_batch_random(folder)
        else:
            print('User Input was not recognized: ' + answer)
    
    def _prompt_training_batch_even(self, folder):
        '''
        Prompts the user with the information needed to 
        perform even batch training.
        '''
        question = 'How many neural nets are in this folder?\n'
        net_num = eval(self._ask(question))
        
        if not os.path.isfile(folder + str(net_num - 1) + '.json'):
            print('Not enough files exist.')
            return
        
        question = 'How many times will each neural net play against each other?\n'
        iterations = eval(self._ask(question))
        
        f = math.factorial
        matches = (f(net_num) // f(2) // f(net_num - 2)) * iterations
        
        question = 'You will train on ' + str(matches) + ' matches. Is this okay? [Y/n]\n'
        answer = self._ask(question)
        if answer == 'n':
            print('Canceled')
            return
        
        question = 'Should they play their best [b] or be somewhat random [r]?\n'
        type = self._ask(question)
        
        print('Training...')
        for x in range(net_num):
            for o in range(net_num):
                if x != o :
                    self.p1 = NN_Player(self.X, folder + str(x) + '.json', type)
                    self.p2 = NN_Player(self.O, folder + str(o) + '.json', type)
                    for i in range(iterations):
                        self._single_game(False)
                        if i % 100 == 0 :
                            self._swap_teams()
                    self.p1.close()
                    self.p2.close()
    
    def _prompt_training_batch_random(self, folder):
        '''
        Prompts the user with the information needed to 
        perform randomly matched batch training.
        '''
        question = 'How many neural nets are in this folder?\n'
        net_num = eval(self._ask(question))
        
        if not os.path.isfile(folder + str(net_num - 1) + '.json'):
            print('Not enough files exist.')
            return
        
        question = 'How many pairs of neural nets will play against each other?\n'
        pairs = eval(self._ask(question))
        
        question = 'How many times will each pair play against each other?\n'
        iterations = eval(self._ask(question))
        
        matches = pairs * iterations
        question = 'You will train on ' + str(matches) + ' matches. Is this okay? [Y/n]\n'
        answer = self._ask(question)
        if answer == 'n':
            print('Canceled')
            return
        
        question = 'Should they play their best [b] or be somewhat random [r]?\n'
        type = self._ask(question)
        
        print('Training...')
        for p in range(pairs):
            r1 = random.randint(0, net_num-1)
            r2 = random.randint(0, net_num-1)
            while r1 == r2 :
                r2 = random.randint(0, net_num-1)
            self.p1 = NN_Player(self.X, folder + str(r1) + '.json', type)
            self.p2 = NN_Player(self.O, folder + str(r2) + '.json', type)
            for i in range(iterations):
                self._single_game(False)
                if i % 100 == 0 :
                    self._swap_teams()
            self.p1.close()
            self.p2.close()
    
    def _prompt_training_pair(self):
        '''
        Prompts the user with the information for training a 
        pair of neural nets.
        '''
        self.p1 = NN_Player(self.X)
        self.p2 = NN_Player(self.O)
        question = 'How many times would you like to train the pair?\n'
        iterations = eval(self._ask(question))
        print('Training...')
        for i in range(iterations):
            self._single_game(False)
            if i % 100 == 0 :
                self._swap_teams()
        self.p1.close()
        self.p2.close()
    
    def _single_game(self, verbose):
        '''
        Runs a single game of TTT. Assumes the players have 
        properly been set. Returns a team integer for the 
        winner.
        '''
        self.board = TTT()
        hw = False
        if verbose:
            print(self.board)
        while not self.board.is_full() and not hw :
            conf, is_valid = self._make_move()
            if not is_valid and verbose:
                print('Not a valid move. Try again.')
            if verbose:
                print(self.board)
                if conf != None :
                    self._pp_confidence(conf)
            hw = self.board.has_won() != self.board.B
        if hw:
            winner = self.board.has_won()
        else:
            winner = self.board.B
        if winner == self.board.X :
            p = 'X won!'
            self.p1.reward()
            self.p2.punish()
        elif winner == self.board.O :
            p = 'O won!'
            self.p2.reward()
            self.p1.punish()
        else:
            p = 'Draw'
            self.p1.draw()
            self.p2.draw()
        if verbose:
            print(p)
        return winner

    def _swap_teams(self):
        '''
        Swaps the teams of p1 and p2.
        '''
        tmp = self.p1.team
        self.p1.set_team(self.p2.team)
        self.p2.set_team(tmp)
    
main = TTT_Controller()