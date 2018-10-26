from Point import Point

class Game(object):
	F = 0
	M = 1
	G = 2
	O = 3
	
	def __init__(self):
		F = Game.F
		self.path = [
					[F, F, F],
					[F, F, F],
					[F, F, F]
					]
		self.marker = None
		self.goal = None
		self.obstacles = []
		self.SetBasic()
					
	def GetPathAsVector(self):
		output = []
		for r in self.path:
			for c in r:
				output.append(c)
		return output
	
	def GetPath(self, p):
		return self.path[p.r][p.c]
		
	def SetPath(self, p, n):
		self.path[p.r][p.c] = n
	
	def PlaceMarker(self, p):
		if self.marker is None and self.GetPath(p) == Game.F:
			self.marker = p
			self.SetPath(p, Game.M)
			
	def PlaceGoal(self, p):
		if self.goal is None and self.GetPath(p) == Game.F:
			self.goal = p
			self.SetPath(p, Game.G)
	
	def PlaceObstacle(self, p):
		if self.GetPath(p) == Game.F:
			self.obstacles.append(p)
			self.SetPath(p, Game.O)
	
	def IsInBounds(self, p):
		if p.r >= 0 and p.c >= 0 :
			if p.r < len(self.path) and p.c < len(self.path[0]) :
				return True
		return False
		
	def IsClear(self, p):
		s = self.GetPath(p)
		if s == 0 or s == 2 :
			return True
		return False
	
	def up(self):
		p = self.marker
		np = Point(p.r - 1, p.c)
		if self.IsInBounds(np) and self.IsClear(np) :
			self.SetPath(p, Game.F)
			self.SetPath(np, Game.M)
			self.marker = np
	
	def down(self):
		p = self.marker
		np = Point(p.r + 1, p.c)
		if self.IsInBounds(np) and self.IsClear(np) :
			self.SetPath(p, Game.F)
			self.SetPath(np, Game.M)
			self.marker = np
	
	def left(self):
		p = self.marker
		np = Point(p.r, p.c - 1)
		if self.IsInBounds(np) and self.IsClear(np) :
			self.SetPath(p, Game.F)
			self.SetPath(np, Game.M)
			self.marker = np
	
	def right(self):
		p = self.marker
		np = Point(p.r, p.c + 1)
		if self.IsInBounds(np) and self.IsClear(np) :
			self.SetPath(p, Game.F)
			self.SetPath(np, Game.M)
			self.marker = np
			
	def has_won(self):
		if self.marker == self.goal:
			return True
		return False
	
	def __str__(self):
		output = ''
		for r in self.path:
			for c in r:
				output = output + str(c)
			output = output + '\n'
		l = len(output) - 1
		output = output[:l]
		return output
	
	def SetBasic(self):
		self.PlaceMarker(Point(0, 1))
		self.PlaceGoal(Point(2, 1))
		self.PlaceObstacle(Point(1, 1))
		
		
'''
x = Game()
x.PlaceMarker(Point(0, 1))
x.PlaceGoal(Point(2, 1))
x.PlaceObstacle(Point(1, 1))
print(x)
print()
x.Left()
x.Down()
x.Down()
print(x)
print()
print(x.HasWon())
x.Right()
print(x)
print(x.HasWon())
'''