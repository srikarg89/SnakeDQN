import pygame
import game.constants as constants
from game.helpers import add, mult

class Snake:

    def __init__(self):
        self.board_width = constants.BOARD_WIDTH
        self.board_height = constants.BOARD_HEIGHT
        self.reset()
    
    def reset(self):
        self.head = (self.board_width // 2, self.board_height // 2)
        self.body = [add(self.head, mult(constants.WEST, i + 1)) for i in range(constants.SNAKE_INIT_LENGTH)]
        self.direction = constants.EAST
        self.ate = False

    
    def move(self, ate):
        self.ate = ate
        if ate:
            self.body = [self.head] + self.body
        else:
            self.body = [self.head] + self.body[:-1]
        self.head = add(self.head, self.direction)
    
