import theano
import theano.tensor as T
from theano import function
from random import random
import numpy as np

class Neural_Net(object):
	
	def __init__(self, layers):
		'''
		layers is a list of the number of nodes per layer in the neural net.
		The first is the number of input nodes. The last is the number of output nodes.
		The number in between are hidden layers. Minimum of 1 hidden layer.
		'''
		self.layers = layers
		self.learning_rate = 0.01
		
		self.init_variables()
		self.init_functions()
		self.generate_all_layers()
	
	def collapse(self, output):
		'''
		Randomly collapse the original output into a single index. The result is 
		a vector of the same length filled with all 0's and a single 1.
		'''
		total = 0.
		for i in output:
			total = total + i
		r = np.random.rand() * total
		for i in range(0, len(output)):
			out = i
			if r < output[i] :
				break
			r = r - output[i]
		move_made = []
		for i in range(len(output)):
			move_made.append(0.)
		move_made[out] = 1.
		return move_made
	
	def generate_all_layers(self):
		'''
		Generates all weights and biases based off of layer list.
		'''
		self.W = []
		self.WP = []
		self.WPblank = []
		self.B = []
		self.BP = []
		self.BPblank = []
		for l in range(1, len(self.layers)):
			w, wp, b, bp = self.generate_single_layer(self.layers[l-1], self.layers[l])
			self.W.append(w)
			self.WP.append(wp)
			self.WPblank.append(wp)
			self.B.append(b)
			self.BP.append(bp)
			self.BPblank.append(bp)
	
	def generate_single_layer(self, input_num, output_num):
		'''
		Returns a weight matrix and bias vector corresponding to 
		the input and output numers.
		'''
		W = []
		WP = []
		for i in range(input_num):
			temp = []
			temp2 = []
			for o in range(output_num):
				r = random()
				temp.append(r*2-1)#possible replacement of 'r'
				temp2.append(0.)
			W.append(np.array(temp))
			WP.append(np.array(temp2))
		B = []
		BP = []
		for o in range(output_num):
			B.append(1.)
			BP.append(0.)
		return W, WP, B, BP
	
	def init_functions(self):
		'''
		Generates a prediction function and a gradient function.
		'''
		self.predict_inputs = [self.inputs]
		self.predict_inputs.extend(self.weights)
		self.predict_inputs.extend(self.biases)
		self.predict_function = function(self.predict_inputs, self.outputs)
		
		self.grad_inputs = self.predict_inputs
		self.grad_inputs.append(self.desired_outputs)
		self.grad_function = function(self.grad_inputs, self.gradient_equation)
	
	def init_variables(self):
		'''
		Symbolicly representing the neural net in theano variables.
		'''
		#Build variables:
		self.inputs = T.vector('input_layer')
		self.weights = []
		self.biases = []
		self.hiddens = []
		for l in range(0, len(self.layers)-1):
			w = 'weights_layer_' + str(l)
			self.weights.append(T.matrix(w))
			b = 'bias_layer_' + str(l)
			self.biases.append(T.vector(b))
		
		#Build equations
		self.hiddens.append(T.nnet.sigmoid(T.dot(self.inputs, self.weights[0]) + self.biases[0]))
		
		for l in range(1, len(self.weights)):
			self.hiddens.append(T.nnet.sigmoid(T.dot(self.hiddens[l-1], self.weights[l]) + self.biases[l]))
		
		self.outputs = T.nnet.sigmoid(T.dot(self.hiddens[len(self.hiddens)-1], self.weights[len(self.weights)-1]) + self.biases[len(self.biases)-1])
		
		#Build gradient variables and equations:
		self.desired_outputs = T.vector('desired_outputs')
		self.cost_equation = -(self.desired_outputs * T.log(self.outputs) + (1-self.desired_outputs) * T.log(1-self.outputs)).sum()
		
		#Create list of all adjustable variables to take gradient with respect to:
		self.adjustables = []
		self.adjustables.extend(self.weights)
		self.adjustables.extend(self.biases)
		
		#Define gradient:
		self.gradient_equation = T.grad(self.cost_equation, self.adjustables)
		
	def predict(self, inputs):
		'''
		Computes an output with the neural net from the inputs given.
		'''
		predict_inputs = [inputs]
		predict_inputs.extend(self.W)
		predict_inputs.extend(self.B)
		output = self.predict_function(*predict_inputs)
		return output