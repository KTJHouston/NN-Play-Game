import theano
import theano.tensor as T
from theano import function
from random import random
import numpy as np

class Neural_Net(object):
	
	def __init__(self, layers, learning_rate):
		'''
		layers is a list of the number of nodes per layer in the neural net.
		The first is the number of input nodes. The last is the number of output nodes.
		The number in between are hidden layers. Minimum of 1 hidden layer.
		'''
		self.layers = layers
		self.learning_rate = learning_rate
		
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
		choice = []
		for i in range(len(output)):
			choice.append(0.)
		choice[out] = 1.
		confidence = output[out] / total
		return choice, confidence
	
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
		B = 1.
		BP = 0.
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
			self.biases.append(T.scalar(b))
		
		#Build equations
		self.hiddens.append(T.nnet.sigmoid(T.dot(self.inputs, self.weights[0]) + self.biases[0]))
		
		for l in range(1, len(self.weights)-1):
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
		return self.predict_function(*predict_inputs)
	
	def train(self, inputs):
		'''
		Computes a collapsed output with the neural net from the inputes given. 
		Then, saves the gradient changes based on the collapsed output. These 
		changes can be rewarded by calling the 'reward' function.
		'''
		p = self.predict(inputs)
		c, confidence = self.collapse(p)
		g = self.gradient(inputs, c)
		self.update(g)
		return c, confidence
	
	def gradient(self, input, desired_output):
		'''
		Computes the gradient to adjust weights and biases, based on the 
		input and desired_output given.
		'''
		grad_inputs = [input]
		grad_inputs.extend(self.W)
		grad_inputs.extend(self.B)
		grad_inputs.append(desired_output)
		return self.grad_function(*grad_inputs)
	
	def update(self, gradient):
		'''
		Saves the gradient to the list of changes to eventually be 
		applied to the weights and biases of the neural net.
		'''
		g = 0
		for w in range(0, len(self.WP)):
			self.WP[w] = self.WP[w] - self.learning_rate * gradient[g]
			g = g + 1
		for b in range(0, len(self.BP)):
			self.BP[b] = self.BP[b] - self.learning_rate * gradient[g]
			g = g + 1
	
	def reward(self, is_positive):
		'''
		Applies the list of changes to the neural net, either as 
		positive or negative reinforcement based on the boolean 
		value given.
		'''
		if is_positive:
			p = 1
		else:
			p = -1
		for w in range(0, len(self.W)):
			self.W[w] = self.W[w] + (p * self.WP[w])
		for b in range(0, len(self.B)):
			self.B[b] = self.B[b] + (p * self.BP[b])
		for w in range(0, len(self.WP)):
			self.WP[w] = self.WPblank[w]
		for b in range(0, len(self.BP)):
			self.BP[b] = self.BPblank[b]
	
	def __str__(self):
		out = ''
		for i in range(0, len(self.W)):
			out = out + str(self.W[i]) + '\n'
			out = out + str(self.B[i]) + '\n'
		return out