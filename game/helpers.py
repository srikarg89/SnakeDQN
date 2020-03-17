import math
import random
import game.constants as constants
from game.constants import NORTH, EAST, SOUTH, WEST, NORTHEAST, NORTHWEST, SOUTHEAST, SOUTHWEST

def add(a, b):
    return (a[0] + b[0], a[1] + b[1])

def mult(a, s):
    return (a[0] * s, a[1] * s)

def sub(a, b):
    return (b[0] - a[0], b[1] - a[1])

def dist(a, b):
    return math.abs(a[0] - b[0]) + math.abs(a[1] - b[1])

def opp(dir):
    return (-dir[0], -dir[1])

def gen_random_loc():
    width, height = constants.BOARD_WIDTH, constants.BOARD_HEIGHT
    return (random.randint(1, width - 2), random.randint(1, height - 2))

def rotate_clockwise(dir):
    mapping = {NORTH: NORTHEAST, NORTHEAST: EAST, EAST: SOUTHEAST, SOUTHEAST: SOUTH, SOUTH: SOUTHWEST, SOUTHWEST: WEST, WEST: NORTHWEST, NORTHWEST: NORTH}
    return mapping[dir]

def rotate_counter_clockwise(dir):
    for i in range(7):
        dir = rotate_clockwise(dir)
    return dir
