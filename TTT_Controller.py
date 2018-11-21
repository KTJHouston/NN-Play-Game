from Neural_Net import Neural_Net
from Tic_Tac_Toe import Tic_Tac_Toe as TTT
from Player import Player, NN_Player, Human_Player

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
                self.p2 = NN_Player(O)
                self.p1 = Human_Player(X)
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
                played_game = False
                #TODO implement create nns
            elif input == 'q' :
                break
            else:
                print('User Input was not recognized.')
    
    def _prompt_training(self):
        question = 'Would you like to train a pair [p] or batch [b]?\n'
        answer = self._ask(question)
        if answer == 'p' :
            self._prompt_pair_training()
        elif answer == 'b' :
            a = 2
            #TODO implement batch training
        else:
            print('User Input was not recognized.')
    
    def _prompt_pair_training(self):
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
            self.p1.reward()
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
        #ptmp = self.p1
        #self.p1 = self.p2
        #self.p2 = ptmp
    
main = TTT_Controller()