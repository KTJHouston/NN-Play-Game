from Neural_Net import Neural_Net
from random import random, randint

#Create test cases:
def create_greatest(input_num):
	'''
	Creates an input list of length input_num of integers 1 through input_num, 
	randomly ordered. The output list is the same length with a 1 at the index
	corresponding to the index of the largest number in the first list.
	'''
	list = []
	inp = []
	out = []
	for i in range(1, input_num+1):
		list.append(i)
	for i in range(0, len(list)):
		r = randint(0, len(list)-1)
		val = list[r]
		inp.append(val)
		if val == input_num :
			out.append(1.)
		else:
			out.append(0.)
		del list[r]
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

def train(nn, iterations, training_function, group_size=1):
	wrong = []
	for i in range(iterations):
		x, d = training_function(nn.layers[0])
		c, conf = nn.train(x)
		if c != d :
			wrong.append(i)
		if i % group_size == 0 :
			if len(wrong) <= (group_size // 2):
				nn.reward(1.)
			else:
				nn.reward(-.25)
			wrong = []
	nn.reward_clear()
	return nn

def test(nn, iterations, training_function):
	for i in range(iterations):
		x, d = training_function(nn.layers[0])
		print(x)
		p = nn.predict(x)
		#print(p)
		c, confidence = nn.collapse(p)
		print(c)
		print(d)
		print('Confidence:')
		pp(confidence)
		if c == d :
			print('CORRECT')
		else:
			print('WRONG')
		print()




'''
layers = [4, 6, 4]
learning_rate = 0.01
NN = Neural_Net(layers, learning_rate)
'''
'''
filename = 'Good_Greatest.json'
NN = Neural_Net(filename=filename)
layers = NN.layers
NN.learning_rate = .001

test_size = 1 #Should be odd
total_wrong = []
wrong = []
for i in range(500000):
	x, d = create_greatest(layers[0])
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
			NN.reward(-.25)
			#NN.reward_clear()
		wrong = []

#Show results:
for i in range(10):
	x, d = create_greatest(layers[0])
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

#Save further trained net:
NN.save(filename)
'''