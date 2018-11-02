from Neural_Net import Neural_Net
from Tic_Tac_Toe import Tic_Tac_Toe as TTT
from Point import Point

class TTT_Wrapper(object):
    
    def __init__(self, nnX, nnO):
        '''
        Takes two neural nets or neural net files as input 
        to play a game of Tic Tac Toe. The layers of the neural 
        nets must begin and end with 9 nodes.
        '''
        if isinstance(nnX, str):
            nnX = Neural_Net(filename=nnX)
        if nnX.layers[0] != 9 or nnX.layers[len(nnX.layers)-1] != 9 :
            raise Exception("Given neural nets must start and end with 9 nodes.")
        self.nnX = nnX

        if isinstance(nnO, str):
            nnO = Neural_Net(filename=nnO)
        if nnO.layers[0] != 9 or nnO.layers[len(nnO.layers)-1] != 9 :
            raise Exception("Given neural nets must start and end with 9 nodes.")
        self.nnO = nnO

        self.board = TTT()
        self.turn = self.board.X
	
    def run(self, max_chances=1, verbose=False):
        '''
        Runs through a game of tic tac toe. max_chances 
        allows the neural net to have multiple 
        chances to make a valid move before having its 
        turn skipped. Returns True if it has won. 
        False otherwise.
        '''
        if verbose:
            print(self.board)
            print()
        hw = False
        while not self.board.is_full() and not hw:
            hw, conf = self._move_once(max_chances)
            if verbose:
                self.pp(conf)
                print(self.board)
                print()
        if hw:
            winner = self.turn
        else:
            winner = self.board.B
        if verbose:
            if winner == self.board.X :
                print('X won!')
            elif winner == self.board.O :
                print('O won!')
            else:
                print('Draw')
        self.board.clear()
        return winner
    
    def train(self, iterations, max_chances):
        '''
        Runs the neural net on the board iterations number 
        of times, limiting the neural nets to max_chances 
        number of attempts to make a valid move. Returns 
        the percentage of [O_wins, draws, X_wins].
        '''
        wins = [0, 0, 0] #[O_wins, draws, X_wins]
        for i in range(iterations):
            w = self.run(max_chances)
            wins[w+1] = wins[w+1] + 1
            if w == self.board.X :
                self.nnX.reward(1.)
                self.nnO.reward(-.05)
            elif w == self.board.O :
                self.nnO.reward(1.)
                self.nnX.reward(-.05)
            else:
                self.nnO.reward(-.05)
                self.nnX.reward(-.05)
        for i in range(len(wins)):
            wins[i] = wins[i] / iterations
        return wins
    
    def test(self, iterations, max_chances):
        '''
        Runs the neural net on the board iterations number 
        of times, calculating a percentage of how often 
        the neural net wins. But, does not reward based on
        the wins.
        '''
        wins = [0, 0, 0] #[O_wins, draws, X_wins]
        for i in range(iterations):
            w = self.run(max_chances)
            wins[w+1] = wins[w+1] + 1
            self.nnX.reward(0.)
            self.nnO.reward(0.)
        for i in range(len(wins)):
            wins[i] = wins[i] / iterations
        return wins
    
    def pp(self, flist):
        '''
        Prints a list of floats, but restricts the number 
        of decimal places to 2.
        '''
        print('[', end='')
        for f in range(len(flist)-1):
            print(f'%.2f, ' % flist[f], end='')
        print(f'%.2f]' % flist[len(flist)-1])
    
    def print_NN(self):
        '''
        Prints the weights and bases in the neural net.
        '''
        print(self.nn)
    
    def demo(self, max_chances):
        '''
        Runs the neural nets through a game once with the 
        verbose setting. 
        '''
        self.run(max_chances, True)
        self.nnX.reward(0.)
        self.nnO.reward(0.)
    
    def _alternate_turn(self):
        '''
        Switches the turn to the other neural net.
        '''
        self.turn = self.turn * -1
    
    def _apply_move(self, output_vector):
        '''
        Places the marker at the point based 
        on the output_vector for the neural net 
        whose turn it is. Only one value in the 
        output_vector should be 1. All others 
        should be zero. Returns true if it was a 
        valid move. False otherwise.
        '''
        for i in range(len(output_vector)):
            if output_vector[i] == 1 :
                pos = i
                break;
        r = 0
        while pos >= 3 :
            r = r + 1
            pos = pos - 3
        point = Point(r, pos)
        if self.turn == self.board.X :
            return self.board.placeX(point)
        else:
            return self.board.placeO(point)
	
    def _move_once(self, max_chances=1):
        '''
        Calculates the next move through the neural net 
        whose turn it is, then applies that move. 
        max_chances allows the neural net to have multiple 
        chances to make a valid move before having its 
        turn skipped. Returns True if that player 
        has won. False, otherwise. Also, returns the 
        confidence level of the move.
        '''
        max_chances = max_chances - 1
        input = self.board.as_vector(self.turn)
        if self.turn == self.board.X :
            output, conf = self.nnX.train(input)
        else:
            output, conf = self.nnO.train(input)
        is_valid = self._apply_move(output)
        if not is_valid and max_chances > 0:
            return self._move_once(max_chances)
        if self.board.has_won() != self.board.B :
            return True, conf
        self._alternate_turn()
        return False, conf
    
def demo_new(layersX, learning_rateX, filenameX, layersO, learning_rateO, filenameO, max_chances=1):
    '''
    Plays a game of tic tac toe with two, newly 
    created neural nets, then saves the neural nets 
    to their respective files. 
    '''
    nnX = Neural_Net(layersX, learning_rateX)
    nnO = Neural_Net(layersO, learning_rateO)
    tttw = TTT_Wrapper(nnX, nnO)
    if max_chances > 0 :
        tttw.demo(max_chances)
    tttw.nnX.save(filenameX)
    tttw.nnO.save(filenameO)

def demo(filenameX, filenameO, max_chances=1):
    '''
    Runs the given neural nets through a game 
    with the verbose flag on.
    '''
    nnX = Neural_Net(filename=filenameX)
    nnO = Neural_Net(filename=filenameO)
    tttw = TTT_Wrapper(nnX, nnO)
    tttw.demo(max_chances)
    
def train_file(filenameX, filenameO, iterations, max_chances=1):
    '''
    Reads in neural nets from the files given.
    Then, trains the neural nets on games played 
    iterations number of times. Overwrites the 
    original files.
    '''
    nnX = Neural_Net(filename=filenameX)
    nnO = Neural_Net(filename=filenameO)
    tttw = TTT_Wrapper(nnX, nnO)
    percent_correct = tttw.train(iterations, max_chances)
    tttw.pp(percent_correct)
    tttw.nnX.save(filenameX)
    tttw.nnO.save(filenameO)

def test_file(filenameX, filenameO, iterations, max_chances=1):
    '''
    Reads in two neural nets from the files given. Then, 
    Runs the neural nets on random boards iterations 
    number of times, calculating a percentage for how 
    often the neural net wins.
    '''
    nnX = Neural_Net(filename=filenameX)
    nnO = Neural_Net(filename=filenameO)
    tttw = TTT_Wrapper(nnX, nnO)
    percent_correct = tttw.test(iterations, max_chances)
    tttw.pp(percent_correct)

fileX = 'Saved_Neural_Nets/TTT/X1.json'
fileO = 'Saved_Neural_Nets/TTT/O1.json'
#demo_new([9, 9, 9], 0.01, fileX, [9, 11, 9], 0.01, fileO)
#train_file(fileX, fileO, 100000)
test_file(fileX, fileO, 10000)
demo(fileX, fileO)
