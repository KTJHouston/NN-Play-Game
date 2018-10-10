class WB(object):
	
	def __init__(self, layers):
		self.weight_matrices = []
		self.biases = []
		for l in range(len(layers)):
			self.weight_matrices.append([])
			self.biases.append(1.)
	
	def get_bias(self, layer):
		return self.biases[layer]
	
	def get_weight(self, layer):
		return self.weight_matrices[layer]
	
	def set_bias(self, layer, value):
		self.biases[layer] = value
	
	def set_weight(self, layer, matrix):
		self.weight_matrices[layer] = matrix