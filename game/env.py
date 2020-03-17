import game.constants as constants
from game.helpers import gen_random_loc

class Environment:

    def __init__(self, snake, display):
        self.board_width = constants.BOARD_WIDTH
        self.board_height = constants.BOARD_HEIGHT
        self.snake = snake
        self.apple = self.generate_apple()
        self.display = display
        self.game_over = False
        self.wall = []
        self.timer = 0
        self.timer_threshold = constants.APPLE_TIMER
        for i in range(self.board_width):
            self.wall.append((i, 0))
            self.wall.append((i, self.board_height - 1))
        for i in range(self.board_height):
            self.wall.append((0, i))
            self.wall.append((self.board_width - 1, i))
    

    def generate_apple(self):
        avoid = [self.snake.head] + self.snake.body
        loc = gen_random_loc()
        while loc in avoid:
            loc = gen_random_loc()
        return loc
        

    def end_game(self):
        self.display.terminate()
        self.snake.terminate(self)
        self.game_over = True


    def check_collision(self):
        if self.snake.head in self.snake.body:
            return True
        if self.snake.head[0] in (0, self.board_width - 1) or self.snake.head[1] in (0, self.board_height - 1):
            return True
        return False


    def check_eaten(self):
        return self.snake.head == self.apple


    def update(self):
#        print("Running")
        self.timer += 1
        if self.timer >= self.timer_threshold:
            self.end_game()
            return

        action = self.snake.act(self)
        if action is None:
            self.end_game()
            return
        self.snake.direction = action
        
        eaten = self.check_eaten()
        if eaten:
            self.apple = self.generate_apple()
            self.timer = 0
        self.snake.move(eaten)

        if self.check_collision():
            self.end_game()
            return

        self.snake.save(self)
        self.display.draw(self.snake, self.apple)


    def run(self):
        self.snake.reset()
#        print("Beginning game")
        while not self.game_over:
            self.update()
            self.display.render()
