from Point import Point

class Tic_Tac_Toe(object):
    X = 1
    O = -1
    B = 0
    
    def __init__(self):
        '''
        Generates a blank board
        '''
        self.clear()
    
    def __str__(self):
        '''
        Returns a string 2D grid representing the board.
        '''
        output = ''
        for r in self.board :
            output = output + '['
            for c in r:
                if c == self.B :
                    t = ' '
                elif c == self.X :
                    t = 'X'
                elif c == self.O :
                    t = 'O'
                output = output + t + ' '
            output = output + ']\n'
        return output
        
    def clear(self):
        '''
        Sets the board to all blank values.
        '''
        B = self.B
        self.board = [
                      [B, B, B],
                      [B, B, B],
                      [B, B, B]
                     ]
            
    def get(self, point):
        '''
        Returns the value at that point on the board, 
        without any safety checks.
        '''
        return self.board[point.r][point.c]
    
    def has_won(self):
        '''
        Returns X if X has won and O if O has won. 
        Returns B if neither have won.
        '''
        X = self.X
        O = self.O
        for r in self.board:
            if r == [X, X, X] :
                return X
            if r == [O, O, O] :
                return O
        for c in range(3):
            col = [
                   self.board[0][c],
                   self.board[1][c],
                   self.board[2][c],
                  ]
            if col == [X, X, X] :
                return X
            if col == [O, O, O] :
                return O
        diag1 = []
        diag2 = []
        for d in range(3):
            diag1.append(self.board[d][d])
            diag2.append(self.board[d][2-d])
        if diag1 == [X, X, X] or diag2 == [X, X, X] :
            return X
        if diag1 == [O, O, O] or diag2 == [O, O, O] :
            return O
        return self.B
    
    def is_clear(self, point):
        '''
        Returns True if the point is within the bounds 
        of the board and the point has a 
        value of B or 0.
        '''
        if self.is_in_bounds(point):
            if self.get(point) == self.B :
                return True
        return False
    
    def is_in_bounds(self, point):
        '''
        Returns True if the point is within the bounds 
        of the board. 
        '''
        if 0 <= point.r and point.r < 3 :
            if 0 <= point.c and point.c < 3 :
                return True
        return False
    
    def place(self, value, point):
        '''
        Places the value at the point on the board.
        '''
        if self.is_clear(point):
            self.set(value, point)
            return True
        return False
    
    def placeX(self, point):
        '''
        Places an X value at the point on the board.
        '''
        self.place(self.X, point)
    
    def placeO(self, point):
        '''
        Places an O value at the point on the board.
        '''
        self.place(self.O, point)
    
    def set(self, value, point):
        '''
        Sets the value at that point on the board, 
        without any safety checks.
        '''
        self.board[point.r][point.c] = value