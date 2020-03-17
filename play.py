import sys
from game.env import Environment
from game.display import Display

if len(sys.argv) > 1:
    filename = sys.argv[1]
    print(filename)
    from players.ai_single import AI as Snake
    snake = Snake(filename)
else:
    from players.human import Human as Snake
    snake = Snake()

test = Environment(snake, Display())
test.run(True)