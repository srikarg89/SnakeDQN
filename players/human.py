import pygame
from players.snake import Snake
import game.constants as constants

class Human(Snake):

    def __init__(self, filename=None):
        super().__init__()
    
    def get_state(self, env):
        return None

    def act(self, state, validate):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None, None
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    return 0, constants.EAST
                elif event.key == pygame.K_LEFT:
                    return 0, constants.WEST
                elif event.key == pygame.K_UP:
                    return 0, constants.NORTH
                elif event.key == pygame.K_DOWN:
                    return 0, constants.SOUTH

        return 0, self.direction
    
    def remember(self, s, a, r, s2):
        return

    def terminate(self, state, action, validate):
        print("Length:", len(self.body) + 1)
