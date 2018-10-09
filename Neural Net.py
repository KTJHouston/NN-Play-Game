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
	
	def __init__(self, L):
		self.layers = L
		self.initMath()
		self.resetGame()
		self.resetMoveOutputs()
	
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
	
	def initMath(self):
		#Define variables:
		x = T.vector('x')#9 input nodes
		w1 = []
		b1 = []
		for h in range(L[1]):#number of hidden nodes
			temp = []
			for v in range(L[0]):#number of input nodes
				temp.append(random())
			w1.append(theano.shared(np.array(temp)))
			b1.append(theano.shared(1.))
		w2 = []
		b2 = []
		for h in range(L[2]):#number of output nodes
			temp = []
			for v in range(L[1]):#number of hidden nodes
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
		
	def move(self):
		'''
		Given the state of the game, calculate a move to change the state.
		Execute that move.
		'''
		inputs = self.game.GetPathAsVector()
		output = self.f(inputs)
		m = self.chooseMove(output)
		if m == self.UP :
			self.game.up()
		elif m == self.DOWN :
			self.game.down()
		elif m == self.LEFT :
			self.game.left()
		elif m == self.RIGHT :
			self.game.right()
	
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
		
	def trainOne(self, moveLimit):
		self.resetGame()
		moveCount = 0
		while not self.game.HasWon() and moveCount < moveLimit :
			self.move()
			moveCount = moveCount + 1
			#Comment out following for speed:
			print(self.game, end='\n\n')
			time.sleep(1)
		if self.game.HasWon() :
			print("WIN!!")
		else :
			print("Loss")
		print(self.game)

L = [9, 9, 4]
nn = NN(L)
nn.trainOne(4)