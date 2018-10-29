import Point
from random import randint

class Maze(object):
	FREE = 0
	MARKER = 1
	GOAL = 2
	OBSTACLE = 3
	
	def __init__(self):
		'''
		Generates a blank 3 by 3 of all free spaces. 
		To populate the map, call generate_basic or 
		generate_random.
		'''
		self.clear()
	
	def __str__(self):
		'''
		Returns a string representation of the map.
		'''
		output = ''
		for r in self.map:
			for c in r:
				output = output + str(c)
			output = output + '\n'
		l = len(output) - 1
		output = output[:l]
		return output
	
	def clear(self):
		'''
		Clears the map of all markers, goals, and obstacles.
		'''
		F = Maze.FREE
		self.map = [
					[F, F, F],
					[F, F, F],
					[F, F, F]
					]
		self.marker = None
		self.goal = None
		self.obstacles = []
	
	def down(self):
		'''
		Moves the marker down, if possible. Otherwise, no change.
		'''
		p = self.marker
		np = Point(p.r + 1, p.c)
		if self.is_in_bounds(np) and self.is_clear(np) :
			self.set_tile(p, Maze.FREE)
			self.set_tile(np, Maze.MARKER)
			self.marker = np
	
	def generate_basic(self):
		'''
		Generates a basic layout for the map. Generation 
		is the same every time.
		'''
		self.clear()
		self.place_marker(Point(0, 1))
		self.place_goal(Point(2, 1))
		self.place_obstacle(Point(1, 1))
	
	def generate_random(self, o=1):
		'''
		Generates a random layout for the map, with o 
		number of obstacles where o <= 7. Does not gaurentee 
		the maze is solvable.
		'''
		self.clear()
		rft = self.random_free_tile
		for i in range(o):
			self.place_obstacle(rft())
		self.place_marker(rft())
		self.place_goal(rft())
		
	def get_map_as_vector(self):
		'''
		Takes the two dimensional list that is the map, 
		and builds a one dimensional list for use as a 
		neural network input vector.
		'''
		output = []
		for r in self.map:
			for c in r:
				output.append(c)
		return output
	
	def get_tile(self, p):
		'''
		Returns the value of the tile at point p.
		'''
		return self.map[p.r][p.c]
			
	def has_won(self):
		'''
		Returns True if marker is on the same tile 
		as goal. False, otherwise.
		'''
		if self.marker == self.goal:
			return True
		return False
		
	def is_clear(self, p):
		'''
		Returns True if the tile at point p can 
		be move into. Meaning, it is either a free 
		tile or a goal tile.
		'''
		s = self.get_tile(p)
		if s == 0 or s == 2 :
			return True
		return False
	
	def is_in_bounds(self, p):
		'''
		Returns True if point p is values 
		0 <= p.r <= len(maze)
		0 <= p.c <= len(maze[0])
		False otherwise.
		'''
		if p.r >= 0 and p.c >= 0 :
			if p.r < len(self.map) and p.c < len(self.map[0]) :
				return True
		return False
	
	def left(self):
		'''
		Moves the marker left, if possible. Otherwise, no change.
		'''
		p = self.marker
		np = Point(p.r, p.c - 1)
		if self.is_in_bounds(np) and self.is_clear(np) :
			self.set_tile(p, Maze.FREE)
			self.set_tile(np, Maze.MARKER)
			self.marker = np
			
	def place_goal(self, p):
		'''
		Places a goal value in the tile at 
		point p. Returns True if successful, 
		False otherwise.
		'''
		if self.goal is None and self.get_tile(p) == Maze.FREE:
			self.goal = p
			self.set_tile(p, Maze.GOAL)
			return True
		return False
	
	def place_marker(self, p):
		'''
		Places a marker value in the tile at 
		point p. Returns True if successful, 
		False otherwise.
		'''
		if self.marker is None and self.get_tile(p) == Maze.FREE:
			self.marker = p
			self.set_tile(p, Maze.MARKER)
			return True
		return False
	
	def place_obstacle(self, p):
		'''
		Places am obstacle value in the tile at 
		point p. Returns True if successful, 
		False otherwise.
		'''
		if self.get_tile(p) == Maze.FREE:
			self.obstacles.append(p)
			self.set_tile(p, Maze.OBSTACLE)
			return True
		return False
		
	def random_free_tile(self):
		'''
		Returns a random point that corresponds to a free 
		tile in the map. If there are no free tiles, returns None.
		'''
		if len(self.obstacles) >= 7 :
			return None
		r = randint(0, len(self.map)-1)
		c = randint(0, len(self.map[0])-1)
		p = Point(r, c)
		while self.get_tile(p) != Maze.FREE :			
			r = randint(0, len(self.map)-1)
			c = randint(0, len(self.map[0])-1)
			p = Point(r, c)
		return p
	
	def right(self):
		'''
		Moves the marker right, if possible. Otherwise, no change.
		'''
		p = self.marker
		np = Point(p.r, p.c + 1)
		if self.is_in_bounds(np) and self.is_clear(np) :
			self.set_tile(p, Maze.FREE)
			self.set_tile(np, Maze.MARKER)
			self.marker = np
		
	def set_tile(self, p, n):
		'''
		Sets the value of the tile at point p.
		'''
		self.map[p.r][p.c] = n
	
	def up(self):
		'''
		Moves the marker up, if possible. Otherwise, no change.
		'''
		p = self.marker
		np = Point(p.r - 1, p.c)
		if self.is_in_bounds(np) and self.is_clear(np) :
			self.set_tile(p, Maze.FREE)
			self.set_tile(np, Maze.MARKER)
			self.marker = np
		
class Point:
	def __init__(self, row, col):
		self.r = row
		self.c = col
		
	def __eq__(self, other):
		if self.r == other.r and self.c == other.c:
			return True
		return False

'''
x = Maze()
x.generate_basic()
print(x)
print()
x.left()
x.down()
x.up()
x.down()
x.down()
print(x)
print(x.has_won())
print()
x.right()
print(x)
print(x.has_won())
'''