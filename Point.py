class Point:
	def __init__(self, row, col):
		self.r = row
		self.c = col
		
	def __eq__(self, other):
		if self.r == other.r and self.c == other.c:
			return True
		return False