#https://towardsdatascience.com/deep-reinforcement-learning-build-a-deep-q-network-dqn-to-play-cartpole-with-tensorflow-2-and-gym-8e105744b998
import os
import numpy as np
from players.actor_critic import AI
from game.env import Environment
from game.display import Display, NoDisplay
import ray

ray.init()

print("Creating snake")
snake = AI()
history = "game/training/multi/"
if not os.path.isdir(history):
    os.mkdir(history)

pool = Pool(processes=4)

print("Starting training")
for i in range(50):
    display = NoDisplay()
    validate = False
#    if i % 1 == -1:
#        print("Displaying: ", i)
#        display = Display()
#        validate = True
    env = Environment(snake, display)
    env.run(validate)
    if i % 10 == 0:
        env.save(history + str(i) + ".txt")

print("Final epsilon:", snake.epsilon)

snake.save_model('multi.h5')
