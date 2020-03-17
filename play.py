from game.env import Environment
from players.human import Human as Snake
from game.display import Display

test = Environment(Snake(), Display())
test.run()