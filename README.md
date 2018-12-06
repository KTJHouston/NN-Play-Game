# Neural-Nets
Working towards a Neural Net to play a board game

In its current state, there is a modular neural network class 
which is being applied to a basic "maze." The maze is only 
intended to be a demonstration of the mathematical functionality 
of the neural network, and not its accuracy. 

The complexity of the neural networks that have been used so far 
have been quadratic (a single hidden layer), and have not been 
particularly successful in accomplishing tasks that with a large
set of inputs and desired outputs (like the maze problem).

The Tic_Tac_Toe branch has the most recent commits but is not 
yet ready for demoing. The main goal is to get neural networks to play 
tic tac toe against each other. To do this, I will be implementing 
an intuitive command line UI and I am creating a "Player" system 
in which both people and neural nets are players. A game of ttt 
can take place between any two players.

In Saved_Neural_Nets/Maze_Solvers/ you will see many .json files 
which are how neural networks are currently stored. Most were 
trained to handle a random maze, but resulted in very little 
success. 'Basic_3.json' was trained on the same basic maze 
only. It can achieve 100% success in 10 moves or about 90% 
success in only 6 moves. 

To show to 'Basic_3.json' in action, run Demo.py in python.