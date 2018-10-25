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
		out.append(0.)
	out[index] = 1.
	return inp, out

def create_binary():
	inp = []
	out = [[1., 0.], [0., 1.]]
	for i in range(0, 2):
		inp.append(randint(0, 10))
	if inp[0] == inp[1] :
		return create_binary()
	elif inp[0] > inp[1] :
		pick = 0
	else: 
		pick = 1
	return inp, out[pick]

def create_addition(input_num, output_num):
	inp = []
	out = []
	total = 0
	max = (output_num - 1) // input_num
	for i in range(input_num):
		r = randint(0, max)
		inp.append(r)
		total = total + r
	for i in range(output_num):
		out.append(0.)
	out[total] = 1.
	return inp, out

def create_AND(r):
	inp = [[0, 0], [0, 1], [1, 0], [1, 1], [1, 1]]
	out = [[0, 1], [0, 1], [0, 1], [1, 0], [1, 0]]
	inp = inp[r]
	out = out[r]
	return inp, out

def pp(flist):
	print('[', end='')
	for f in range(len(flist)-1):
		print(f'%.2f, ' % flist[f], end='')
	print(f'%.2f]' % flist[len(flist)-1])




	
	




'''
layers = [4, 5, 4]
learning_rate = 0.01
NN = Neural_Net(layers, learning_rate)
'''
NN = Neural_Net(filename='Good_Greatest.json')
layers = NN.layers

test_size = 1 #Should be odd
total_wrong = []
wrong = []
low = 2
high = 6
for i in range(100):
	x, d = create_greatest(layers[0], low, high)
	c, conf = NN.train(x)
	if c != d : 
		total_wrong.append(i)
		wrong.append(i)
	if i % test_size == 0 :
		#if True:
		#if len(wrong) <= (test_size // 2):
		if len(wrong) == 0:
			NN.reward(1.)
		else:
			NN.reward_clear()
		wrong = []

'''
#Save Neural Net:
filename = 'savenn.json'
NN.save(filename)
NNloaded = Neural_Net(layers, 1)
NNloaded.load(filename)
print(NN)
print()
print(NNloaded)
print()

#Continue Training:
for i in range(50000):
	x, d = create_greatest(layers[0], low, high)
	c, conf = NNloaded.train(x)
	if c != d : 
		total_wrong.append(i)
		wrong.append(i)
		print('wrong')
	else:
		print('right')
	if i % test_size == 0 :
		#if True:
		#if len(wrong) <= (test_size // 2):
		if len(wrong) == 0:
			NNloaded.reward(1.)
		else:
			NNloaded.reward_clear()
		wrong = []
'''

#Show results:
for i in range(10):
	x, d = create_greatest(layers[0], low, high)
	print(x)
	#print(d)
	p = NN.predict(x)
	print(p)
	c, confidence = NN.collapse(p)
	print(c)
	print(d)
	print('Confidence:')
	pp(confidence)
	if c == d :
		print('CORRECT')
	else:
		print('WRONG')
	print()