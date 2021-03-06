import theano
import theano.tensor as T
from theano import function
from random import random
import json

class Neural_Net(object):
	
	def __init__(self, layers=None, learning_rate=None, filename=None):
		'''
		layers is a list of the number of nodes per layer in the neural net.
		The first is the number of input nodes. The last is the number of output nodes.
		The numbers in between are hidden layers. Minimum of 1 hidden layer. 
		learning_rate is the percentage of the gradient applied to the weights and biases. 
		filename is the name of a file to load a neural network from. Both the layers and 
		learning_rate OR a filename are needed to initialize the function.
		'''
		if filename == None :
			if layers == None or learning_rate == None :
				raise Exception("layers list or learning_rate value not provided")
			self.layers = layers
			self.learning_rate = learning_rate
			
			self.init_variables()
			self.init_functions()
			self.generate_all_layers()
		else:
			self.load(filename)
	
	def __str__(self):
		'''
		String representation of the Neural Net. Lists all the weights and biases 
		that define the Neural Net.
		'''
		out = ''
		for i in range(0, len(self.W)):
			out = out + str(self.W[i]) + '\n'
			out = out + str(self.B[i]) + '\n'
		return out
	
	def collapse(self, output):
		'''
		Randomly collapse the original output into a single index. The result is 
		a vector of the same length filled with all 0's and a single 1.
		'''
		total = 0.
		for i in output:
			total = total + i
		r = random() * total
		for i in range(0, len(output)):
			out = i
			if r < output[i] :
				break
			r = r - output[i]
		choice = []
		for i in range(len(output)):
			choice.append(0.)
		choice[out] = 1.
		confidence = []
		for i in range(len(output)):
			confidence.append(output[i] / total)
		return choice, confidence
	
	def generate_all_layers(self):
		'''
		Generates all weight and bias values based off of layer list.
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
		Returns a weight matrix and bias vector of random values corresponding 
		to the input and output numbers.
		'''
		W = []
		WP = []
		for i in range(input_num):
			temp = []
			temp2 = []
			for o in range(output_num):
				r = random() * 2 - 1
				temp.append(r)
				temp2.append(0.)
			W.append(temp)
			WP.append(temp2)
		B = 1.
		BP = 0.
		return W, WP, B, BP
	
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
	
	def load(self, filename):
		'''
		Loads layers, learning_rate, weights, and biases from the file given.
		'''
		with open(filename, 'r') as read_file:
			data = json.load(read_file)
			self.layers = data['layers']
			self.learning_rate = data['learning_rate']
			
			self.W = []
			self.WP = []
			self.WPblank = []
			for i in range(0, len(data['weights'])):
				self.W.append(data['weights'][str(i)])
				self.WP.append(data['weightsblank'][str(i)])
				self.WPblank.append(data['weightsblank'][str(i)])
			
			self.B = []
			self.BP = []
			self.BPblank = []
			for i in range(0, len(data['biases'])):
				self.B.append(data['biases'][str(i)])
				self.BP.append(data['biasesblank'][str(i)])
				self.BPblank.append(data['biasesblank'][str(i)])
			
			read_file.close()
			
			self.init_variables()
			self.init_functions()
		
	def predict(self, inputs):
		'''
		Computes an output with the neural net from the inputs given.
		'''
		predict_inputs = [inputs]
		predict_inputs.extend(self.W)
		predict_inputs.extend(self.B)
		return self.predict_function(*predict_inputs)
	
	def reward(self, strength):
		'''
		Applies the list of changes to the neural net, either as 
		positive or negative reinforcement at a certain percentage 
		of strength based on the boolean value given.
		'''
		for w in range(0, len(self.W)):
			self.W[w] = self.W[w] + (strength * self.WP[w])
		for b in range(0, len(self.B)):
			self.B[b] = self.B[b] + (strength * self.BP[b])
		self.reward_clear()
	
	def reward_clear(self):
		'''
		Resets weight and bias gradient that could be applied to 
		the Neural Net's weights and biases.
		'''
		for w in range(0, len(self.WP)):
			self.WP[w] = self.WPblank[w]
		for b in range(0, len(self.BP)):
			self.BP[b] = self.BPblank[b]
	
	def save(self, filename):
		'''
		Saves layers, learning_rate, weights, and biases to the file given.
		'''
		with open(filename, 'w') as write_file:
			data = {
				'layers': self.layers,
				'learning_rate': self.learning_rate
			}
			
			weights = {}
			weightsblank = {}
			for i in range(0, len(self.W)):
				if isinstance(self.W[i], list):
					weights[i] = self.W[i]
				else:
					weights[i] = self.W[i].tolist()
				weightsblank[i] = self.WPblank[i]
			data['weights'] = weights
			data['weightsblank'] = weightsblank
			
			biases = {}
			biasesblank = {}
			for i in range(0, len(self.B)):
				biases[i] = self.B[i]
				biasesblank[i] = self.BPblank[i]
			data['biases'] = biases
			data['biasesblank'] = biasesblank
			
			json.dump(data, write_file)
			write_file.close()
	
	def train(self, inputs, desired=0):
		'''
		Computes a collapsed output with the neural net from the inputes given. 
		Then, saves the gradient changes based on the collapsed output. These 
		changes can be rewarded by calling the 'reward' function.
		'''
		p = self.predict(inputs)
		c, confidence = self.collapse(p)
		
		if desired == 0 :
			cd = c
		else:
			cd = desired
		g = self.gradient(inputs, cd)
		self.update(g)
		return c, confidence
	
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