import math
import random
import game.constants as constants

def add(a, b):
    return (a[0] + b[0], a[1] + b[1])

def mult(a, s):
    return (a[0] * s, a[1] * s)

def dist(a, b):
    return math.abs(a[0] - b[0]) + math.abs(a[1] - b[1])

def opp(dir):
    return (-dir[0], -dir[1])

def gen_random_loc():
    width, height = constants.BOARD_WIDTH, constants.BOARD_HEIGHT
    return (random.randint(1, width - 2), random.randint(1, height - 2))