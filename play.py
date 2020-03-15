from game.env import Environment
from players.ai import AI as Snake
from game.display import Display

test = Environment(Snake(), Display())
test.run()