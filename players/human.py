import pygame
from players.snake import Snake
import game.constants as constants

class Human(Snake):

    def act(self, env):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    return constants.EAST
                elif event.key == pygame.K_LEFT:
                    return constants.WEST
                elif event.key == pygame.K_UP:
                    return constants.NORTH
                elif event.key == pygame.K_DOWN:
                    return constants.SOUTH

        return self.direction
    
    def save(self, env):
        pass

    def terminate(self, env):
        print("Length:", len(self.body) + 1)
