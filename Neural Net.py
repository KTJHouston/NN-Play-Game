import theano
import theano.tensor as T
from theano.ifelse import ifelse
from theano import function
from theano import shared
import numpy as np
from random import random
from Game import Game

import time

class NN(object):
	UP = 0
	DOWN = 1
	LEFT = 2
	RIGHT = 3
	
	def __init__(self):
		self.initMath()
		self.resetGame()
		self.resetMoveOutputs()
	
	def initMath(self):
		#Define variables:
		x = T.vector('x')#9 input nodes
		w1 = []
		b1 = []
		for h in range(9):#number of hidden nodes
			temp = []
			for v in range(9):#number of input nodes
				temp.append(random())
			w1.append(theano.shared(np.array(temp)))
			b1.append(theano.shared(1.))
		w2 = []
		b2 = []
		for h in range(4):#number of output nodes
			temp = []
			for v in range(9):#number of hidden nodes
				temp.append(random())
			w2.append(theano.shared(np.array(temp)))
			b2.append(theano.shared(1.))
			
		#Define mathematical expression:
		#Layer 1:
		a1 = []
		for i in range(len(w1)):
			a1.append(1/(1+T.exp(-T.dot(x, w1[i])-b1[i])))
		a1combined = []
		for a in a1:
			a1combined.append(a)
		a1combined = T.stack(a1combined, axis=1)
		
		#Output Layer:
		a2 = []
		for i in range(len(w2)):
			a2.append(1/(1+T.exp(-T.dot(a1combined, w2[i])-b2[i])))

		self.f = function([x], a2)
	
	def resetGame(self):
		'''
		Initializes an instance of Game with basic set up.
		'''
		self.game = Game()
		self.game.SetBasic()
	
	def resetMoveOutputs(self):
		'''
		Initialize arrays for saving the input and output
		'''
		self.output = []
		self.move_made = []
		
	def move(self):
		'''
		Given the state of the game, calculate a move to change the state.
		Execute that move.
		'''
		inputs = self.game.GetPathAsVector()
		output = self.f(inputs)
		m = self.chooseMove(output)
		if m == 0 :
			self.game.up()
		elif m == 1 :
			self.game.down()
		elif m == 2 :
			self.game.left()
		elif m == 3 :
			self.game.right()
	
	def chooseMove(self, output):
		'''
		Randomly choose a move from the given output and save the result.
		'''
		total = 0
		for i in output:
			total = total + i
		r = np.random.rand() * total
		for i in range(0, len(output)):
			out = i
			if r < output[i] :
				break
			r = r - output[i]
		move_made = [0, 0, 0, 0]
		move_made[out] = 1
		self.output.append(output)
		self.move_made.append(move_made)
		return out
		
	def trainOne(self, moveLimit):
		self.resetGame()
		moveCount = 0
		while not self.game.HasWon() and moveCount < moveLimit :
			self.move()
			moveCount = moveCount + 1
			print(self.game, end='\n\n')
			time.sleep(1)
		if self.game.HasWon() :
			print("WIN!!")
		else :
			print("Loss")
		print(self.game)

nn = NN(9)
nn.trainOne(4)

'''
#Define variables:
x = T.vector('x')#9 input nodes
w1 = []
b1 = []
for h in range(9):#number of hidden nodes
	temp = []
	for v in range(9):#number of input nodes
		temp.append(random())
	w1.append(theano.shared(np.array(temp)))
	b1.append(theano.shared(1.))
w2 = []
b2 = []
for h in range(4):#number of output nodes
	temp = []
	for v in range(9):#number of hidden nodes
		temp.append(random())
	w2.append(theano.shared(np.array(temp)))
	b2.append(theano.shared(1.))

learning_rate = 0.01

#Define mathematical expression:
#Layer 1:
a1 = []
for i in range(len(w1)):
	a1.append(1/(1+T.exp(-T.dot(x, w1[i])-b1[i])))
a1combined = []
for a in a1:
	a1combined.append(a)
a1combined = T.stack(a1combined, axis=1)

#Output Layer:
a2 = []
for i in range(len(w2)):
	a2.append(1/(1+T.exp(-T.dot(a1combined, w2[i])-b2[i])))

#Define inputs:
g = Game()
g.SetBasic()
inputs = g.GetPathAsVector()
print(inputs)

f = function([x], a2)
print("Test")
print(f(inputs))
'''
'''
#Cost
a_hat = T.vector('a_hat') #actual output
cost = -(a_hat * T.log(a) + (1 - a_hat) * T.log(1-a)).sum()

#Gradients
dw, db = T.grad(cost, [w, b])

train = function(
	inputs = [x, a_hat], 
	outputs = [a, cost], 
	updates = [
		[w, w - learning_rate * dw],
		[b, b - learning_rate * db],
	]
)

#Define inputs and desired outputs
inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
outputs = [0, 0, 0, 1]

#Iterate through all inputs and find outputs:
cost = []
for iteration in range(30000):
    pred, cost_iter = train(inputs, outputs)
    cost.append(cost_iter)
    
#Print the outputs:
print('The outputs of the NN are:')
for i in range(len(inputs)):
    print('The output for x1=%d | x2=%d is %.2f' % (inputs[i][0],inputs[i][1],pred[i]))
   
print('The final weights and biases are:')
print(w.get_value())
print(b.get_value())
'''
'''   
#Plot the flow of cost:
print('\nThe flow of cost during model run is as following:')
import matplotlib.pyplot as plt
%matplotlib inline
plt.plot(cost)
'''

'''
nz = function([x], z)
neuron = function([x], a)

#Define inputs and weights
inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
weights = [1, 1]
bias = -1.5

#Iterate through all inputs and find outputs:
for i in range(len(inputs)):
	t = inputs[i]
	out = neuron(t)
	print ('The output for x1=%d | x2=%d is %d' % (t[0],t[1],out))
	out = nz(t)
	print ('The output for x1=%d | x2=%d is %f' % (t[0],t[1],out))
	print()
'''