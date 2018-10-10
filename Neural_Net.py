import theano
import theano.tensor as T
from theano import function
from random import random
from WB import WB
import numpy as np

class Neural_Net(object):
	
	def __init__(self, layers):
		self.layers = layers
		self.net_values = WB(self.layers)
		self.init_variables()
		self.generate_random()
		self.init_functions()
	
	def generate_random(self):
		w = []
		for h in range(self.layers[0]):
			temp = []
			for i in range(self.layers[1]):
				temp.append(random())
			w.append(np.array(temp))
		self.net_values.set_weight(0, w)
		self.net_values.set_bias(0, 1.)
		w = []
		for o in range(self.layers[1]):
			temp = []
			for h in range(self.layers[2]):
				temp.append(random())
			w.append(np.array(temp))
		self.net_values.set_weight(1, w)
		self.net_values.set_bias(1, 1.)
	
	def init_functions(self):
		self.calculate = function(
			[
				self.input_layer,
				self.weights_layer_0,
				self.bias_layer_0,
				self.weights_layer_1,
				self.bias_layer_1
			],
			self.output_layer
		)
	
	def init_variables(self):
		'''Symbolicly representing the neural net in theano variables.'''
		self.input_layer = T.vector('input_layer') #self.layers[0] number of input nodes
		self.weights_layer_0 = T.matrix('weights_layer_0')
		self.bias_layer_0 = T.scalar('bias_layer_0')
		self.hidden_layer = 1 / (1 + T.exp(-T.dot(self.input_layer, self.weights_layer_0) - self.bias_layer_0))
		
		self.weights_layer_1 = T.matrix('weights_layer_1')
		self.bias_layer_1 = T.scalar('bias_layer_1')
		self.output_layer = 1 / (1 + T.exp(-T.dot(self.hidden_layer, self.weights_layer_1) - self.bias_layer_1))
	
	def predict(self, inputs):
		output = self.calculate(inputs, self.net_values.get_weight(0), self.net_values.get_bias(0), self.net_values.get_weight(1), self.net_values.get_bias(1))
		return output


layers = [9, 9, 4]
NN = Neural_Net(layers)
test = [0, 1, 2, 3, 0, 0, 1, 1, 1]
print(NN.predict(test))