import Maze_Wrapper as MW
import time

def demo(folder, file, max_moves, iterations):
    print('File: ' + file)
    print('Single run through with a maximum of ' + str(max_moves) + ' moves:')
    MW.demo_file(folder+file, max_moves)
    time.sleep(3)
    print('In %d runs' % iterations)
    MW.test_file(folder+file, iterations, max_moves)
    print('within %d moves.' % max_moves)
    time.sleep(3)





print('0 -> Empty Space')
print('1 -> Current Location')
print('2 -> Goal')
print('3 -> Blockade')
print()
print('Each step shows the likelihood it would move in each direction:')
print('[up, down, left, right]')
print()
time.sleep(3)

folder = 'Saved_Neural_Nets/Maze_Solvers/'
file = 'Basic_3.json'
max_moves = 6
iterations = 1000

demo(folder, file, max_moves, iterations)

max_moves = 10

demo(folder, file, max_moves, iterations)