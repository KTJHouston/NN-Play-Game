from Neural_Net import Neural_Net
from Maze import Maze

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
		self.maze.generate_random()
	
	def move_once(self):
		'''
		Calculates next move through the neural net 
		once, and applies that move. Returns True if 
		destination reached. False, otherwise.
		'''
		input = self.maze.get_map_as_vector()
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
		print(self.maze)
		print()
		for i in range(max_moves):
			hw, conf = self.move_once()
			if verbose:
				print(f'%d:' % i)
				self.pp(conf)
				print(self.maze)
				print()
			if hw:
				break
		self.maze.generate_random()
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
		'''
		Prints a list of floats, but restricts the number 
		of decimal places to 2.
		'''
		print('[', end='')
		for f in range(len(flist)-1):
			print(f'%.2f, ' % flist[f], end='')
		print(f'%.2f]' % flist[len(flist)-1])
	
def demo_file(filename, max_moves):
	'''
	Reads in a neural network from the file given, 
	then runs it through the maze once with the 
	verbose setting. 
	'''
	nn = Neural_Net(filename=filename)
	mw = Neural_Net_Maze_Wrapper(nn)
	mw.run(max_moves, True)

def demo_new(layers, learning_rate, filename, max_moves=0):
	'''
	Randomly generates a new neural net, demos 
	it, then saves it to the filename given.
	'''
	nn = Neural_Net(layers, learning_rate)
	mw = Neural_Net_Maze_Wrapper(nn)
	if max_moves > 0 :
		mw.run(max_moves, True)
	mw.nn.save(filename)

def train_file(filename, iterations, max_moves, savefile=None):
	'''
	Reads in a neural network from the file given.
	Then, trains the data based on the number of iterations.
	Saves the file to either another name, if given, 
	or overwriting the original file.
	'''
	nn = Neural_Net(filename=filename)
	mw = Neural_Net_Maze_Wrapper(nn)
	mw.train(iterations, max_moves)
	if savefile == None :
		savefile = filename
	mw.nn.save(savefile)