import pygame
import game.constants as constants

class Display:

    def __init__(self):
        self.board_width = constants.BOARD_WIDTH
        self.board_height = constants.BOARD_HEIGHT
        self.block_size = constants.BLOCK_SIZE
        self.screen = pygame.display.set_mode((self.board_width * self.block_size, self.board_width * self.block_size))
        self.clock = pygame.time.Clock()


    def draw_block(self, color, loc):
        x, y = loc
        pygame.draw.rect(self.screen, color, pygame.Rect(x * self.block_size, y * self.block_size, self.block_size, self.block_size))


    def draw_wall(self):
        color = constants.WALL_COLOR
        for i in range(self.board_width):
            self.draw_block(color, (i, 0))
            self.draw_block(color, (i, self.board_height - 1))
        for i in range(self.board_height):
            self.draw_block(color, (0, i))
            self.draw_block(color, (self.board_width - 1, i))


    def draw(self, snake, apple):
        self.screen.fill(constants.BACKGROUND_COLOR)
        self.draw_wall()
        self.draw_block(constants.APPLE_COLOR, apple)
        snake_locs = [snake.head] + snake.body
        for loc in snake_locs:
            self.draw_block(constants.SNAKE_COLOR, loc)
        pygame.display.flip()
    

    def render(self):
        self.clock.tick(constants.CLOCK_SPEED)


    def terminate(self):
        pygame.quit()



class NoDisplay:

    def draw_block(self):
        pass
    
    def draw_wall(self):
        pass
    
    def draw(self):
        pass
    
    def render(self):
        pass
    
    def terminate(self):
        pass