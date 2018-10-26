from Neural_Net import Neural_Net
from Maze import Game as Maze

class Neural_Net_Maze_Wrapper(object):
	
	def __init__(self, nn):
		'''
		Takes a neural net as to run the maze with. 
		The layers of the neural net must begin with 
		9 and end with 4.
		'''
		la = nn.layers
		if la[0] != 9 or la[len(la)-1] != 4 :
			raise Exception("Given neural net must start with 9 nodes and end with 4.")
		self.nn = nn
		self.maze = Maze()
	
	def run_once(self):
		'''
		Calculates next move through the neural net 
		once, and applies that move. Returns True if 
		destination reached. False, otherwise.
		'''
		input = self.maze.GetPathAsVector()
		output, conf = self.nn.train(input)
		self.apply_move(output)
		return self.maze.has_won(), conf
	
	def apply_move(self, output_vector):
		'''
		Moves the marker in the maze based on the 
		output_vector. Only one value should be 1. 
		All others should be zero. Action: 
		[up, down, left, right]
		'''
		moves = {
				0: self.maze.up,
				1: self.maze.down,
				2: self.maze.left,
				3: self.maze.right
				}
		for i in range(len(output_vector)):
			if output_vector[i] == 1. :
				moves[i]()
	
	def run(self, max_moves, verbose=False):
		'''
		Runs through the maze, making a number of 
		moves up to max_moves, returning True if it 
		has won.
		'''
		for i in range(max_moves):
			hw, conf = self.run_once()
			if verbose:
				self.pp(conf)
				print(f'%d:' % i)
				print(self.maze)
				print()
			if hw:
				break
		self.maze = Maze()
		if hw:
			return True, i
		return False, -1
	
	def train(self, iterations, max_moves):
		'''
		Runs the neural net on the maze iterations number 
		of times, limiting the runs the max_moves number 
		of moves.
		'''
		for i in range(iterations):
			hw = self.run(max_moves)
			if hw[0]:
				self.nn.reward(1.)
			else:
				self.nn.reward(-.25)
	
	def test(self, iterations, max_moves):
		'''
		Runs the neural net on the maze iterations number 
		of times, but does not reward the function based 
		on the output.
		'''
		for i in range(iterations):
			hw = self.run(max_moves)
			print(hw)
			print()
		self.nn.reward_clear()
	
	def pp(self, flist):
		print('[', end='')
		for f in range(len(flist)-1):
			print(f'%.2f, ' % flist[f], end='')
		print(f'%.2f]' % flist[len(flist)-1])

'''
#Train a neural net:
layers = [9, 6, 4]
learning_rate = 0.01
nn = Neural_Net(layers, learning_rate)
mw = Neural_Net_Maze_Wrapper(nn)
mw.train(10000, 15)
mw.nn.save('First.json')

#Test the neural net:
nn = Neural_Net(filename='First.json')
mw = Neural_Net_Maze_Wrapper(nn)
mw.test(5, 15)
'''
#Test the neural net verbose:
nn = Neural_Net(filename='Final.json')
mw = Neural_Net_Maze_Wrapper(nn)
mw.run(15, True)
mw.nn.save('Saved_Neural_Nets/Maze_Solvers/Final.json')
'''
#Further train more constrained neural net:
nn = Neural_Net(filename='Second.json')
mw = Neural_Net_Maze_Wrapper(nn)
mw.train(1000, 4)
mw.nn.save('Final.json')
'''