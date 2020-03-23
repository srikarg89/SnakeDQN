#https://towardsdatascience.com/deep-reinforcement-learning-build-a-deep-q-network-dqn-to-play-cartpole-with-tensorflow-2-and-gym-8e105744b998
import os
import numpy as np
from players.ai_single import AI
from game.env import Environment
from game.display import Display, NoDisplay

snake = AI()
#history = "game/training/hidden50_2000/"
history = "game/training/hidden50_10k/"

for filename in os.listdir(history):
    path = os.path.join(history, filename)
    display = Display()
    display.playback(path)
