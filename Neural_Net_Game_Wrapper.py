from Neural_Net import Neural_Net
from random import random, randint

#Create test cases:
def create_greatest(input_num, low, high):
	'''
	Creates a list of length size of random integers, 
	and another list of length size with a 1 in the index
	corresponding to the index of the largest number in the first list.
	'''
	inp = []
	out = []
	index = -1
	largest = -1
	for i in range(0, input_num):
		r = randint(low, high)
		if r > largest :
			inp.append(r)
			largest = r
			index = i
		elif r == largest :
			inp.append(r+1)
			largest = r+1
			index = i
		else:
			inp.append(r)
		out.append(0)
	out[index] = 1
	return inp, out

def create_addition():
	inp = []
	out = []
	total = 0
	max = output_num // input_num
	for i in range(input_num):
		r = randint(0, max)
		inp.append(r)
		total = total + r
	for i in range(output_num+1):
		out.append(0.)
	out[total] = 1
	return inp, out

layers = [2, 2, 2]
NN = Neural_Net(layers)
x, d = create_greatest(layers[0], 0, 10)
print(x)
p = NN.predict(x)
print(p)
c = NN.collapse(p)
print(c)