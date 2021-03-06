import game.constants as constants
from game.helpers import gen_random_loc, add

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
        

    def end_game(self, state, action, validate):
        self.display.terminate()
        self.snake.terminate(state, action, validate)
        self.game_over = True


    def check_collision(self):
        if self.snake.head in self.snake.body:
            return True
        if self.snake.head[0] in (0, self.board_width - 1) or self.snake.head[1] in (0, self.board_height - 1):
            return True
        return False


    def check_eaten(self):
        return add(self.snake.head, self.snake.direction) == self.apple


    def get_board(self):
        return (self.snake.head, *self.snake.body, self.apple)


    def update(self, validate):
        state = self.snake.get_state(self)
        action, direction = self.snake.act(state, validate)
        if direction is None:
            self.end_game(state, action, validate)
            return
        self.snake.direction = direction
        
        eaten = self.check_eaten()
        if eaten:
            self.apple = self.generate_apple()
            self.timer = 0
        self.snake.move(eaten)

        if self.check_collision():
            self.end_game(state, action, validate)
            return

        self.timer += 1
        if self.timer >= self.timer_threshold:
            self.end_game(state, action, validate)
            return

        board = self.get_board()
        if board in self.visited:
            self.end_game(state, action, validate)
            return
        self.visited.add(board)
        self.game_history.append(board)

        next_state = self.snake.get_state(self)
        if not validate:
            self.snake.remember(state, action, eaten, next_state)
            self.snake.replay()

        self.display.draw(self.snake, self.apple)


    def run(self, validate):
        self.snake.reset()
#        print("Beginning game")
        self.visited = set()
        self.game_history = []
        while not self.game_over:
            self.update(validate)
            self.display.render()

    def save(self, filename):
        file = open(filename, "w+")
        for board in self.game_history:
            file.write(str(board) + "\n")
        file.close()