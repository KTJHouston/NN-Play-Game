from Neural_Net import Neural_Net
from Maze import Maze

class Maze_Wrapper(object):
	
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
		self.maze.generate_basic()
	
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
		if verbose:
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
		self.maze.generate_basic()
		if hw:
			return True, i
		return False, -1
	
	def train(self, iterations, max_moves):
		'''
		Runs the neural net on the maze iterations number 
		of times, limiting the runs the max_moves number 
		of moves.
		'''
		correct = 0
		for i in range(iterations):
			hw = self.run(max_moves)
			if hw[0]:
				self.nn.reward(1.)
				correct = correct + 1
			else:
				self.nn.reward(-.25)
		return correct / iterations
	
	def test(self, iterations, max_moves):
		'''
		Runs the neural net on the maze iterations number 
		of times, calculating a percentage of how often 
		the neural net wins. But, does not reward based on
		the wins.
		'''
		correct = 0
		for i in range(iterations):
			hw = self.run(max_moves)
			if hw[0]:
				correct = correct + 1
		self.nn.reward_clear()
		return correct / iterations
	
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
		print(nn)
	
def demo_file(filename, max_moves):
	'''
	Reads in a neural net from the file given, 
	then runs it through the maze once with the 
	verbose setting. 
	'''
	nn = Neural_Net(filename=filename)
	mw = Maze_Wrapper(nn)
	mw.run(max_moves, True)

def demo_new(layers, learning_rate, filename, max_moves=0):
	'''
	Randomly generates a new neural net, demos 
	it, then saves it to the filename given.
	'''
	nn = Neural_Net(layers, learning_rate)
	mw = Maze_Wrapper(nn)
	if max_moves > 0 :
		mw.run(max_moves, True)
	mw.nn.save(filename)

def train_file(filename, iterations, max_moves, savefile=None):
	'''
	Reads in a neural net from the file given.
	Then, trains the data based on the number of iterations.
	Saves the file to either another name, if given, 
	or overwriting the original file.
	'''
	nn = Neural_Net(filename=filename)
	mw = Maze_Wrapper(nn)
	percent_correct = mw.train(iterations, max_moves)
	print(f'%.2f correct' % percent_correct)
	if savefile == None :
		savefile = filename
	mw.nn.save(savefile)

def test_file(filename, iterations, max_moves):
	'''
	Reads in a neural net from the file given. Then, 
	Runs the neural net on random mazes iterations 
	number of times, calculating a percentage for how 
	often the neural net wins.
	'''
	nn = Neural_Net(filename=filename)
	mw = Maze_Wrapper(nn)
	percent_correct = mw.test(iterations, max_moves)
	print(f'%.2f correct' % percent_correct)

def extensive_train(filename, iterations, max, min, savefile=None):
	'''
	Reads in a neural net from the file given. Then, 
	trains the net iterations number of times with max 
	number of max moves. Then steps down, training again 
	iterations number of times. Process continues through 
	min number of max_moves has been trained. The neural 
	net is then saved at the original file location or 
	at savefile if provided.
	'''
	nn = Neural_Net(filename=filename)
	mw = Maze_Wrapper(nn)
	for r in range(max, min-1, -1):
		percent_correct = mw.train(iterations, r)
		print(f'%d: %.2f correct' % (r, percent_correct))
	if savefile == None :
		savefile = filename
	mw.nn.save(savefile)

def full_train(filename, epochs, iterations, max, min, logfile=None):
	nn = Neural_Net(filename=filename)
	mw = Maze_Wrapper(nn)
	log = ''
	for e in range(epochs):
		for r in range(max, min-1, -1):
			percent_correct = mw.train(iterations, r)
			form = f'%d.%d: %.2f correct' % (e, r, percent_correct)
			print(form)
			log = log + form + '\n'
		print()
		log = log + '\n'
	mw.nn.save(filename)
	with open(logfile, 'w') as write_file:
			write_file.write(log)
			write_file.close()