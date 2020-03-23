#https://towardsdatascience.com/deep-reinforcement-learning-build-a-deep-q-network-dqn-to-play-cartpole-with-tensorflow-2-and-gym-8e105744b998
import numpy as np
from players.ai_single import AI
from game.env import Environment
from game.display import Display, NoDisplay

def get_strength(group):
    evaluations = [
        [0, 1, 0],
        [0.75, 0.75, 0],
        [1, 0, 0],
        [1, 0, 0],
        [0.75, 0, 0.75],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0.75, 0.75],
    ]
    dists = 0
    for idx in range(8):
        e = evaluations[idx]
        g = group[idx]
        dist = sum([abs((g[i] + 1)/2 - e[i]) for i in range(len(g))])
        dists += dist
    return dists


def get_closeness(group):
    evaluations = [
        [0, 1, 0],
        [0.75, 0.75, 0],
        [1, 0, 0],
        [1, 0, 0],
        [0.75, 0, 0.75],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0.75, 0.75],
    ]
    correct = 0
    for idx in range(8):
        e = evaluations[idx]
        g = group[idx]
        if e[np.argmax(g)] != 0:
            correct += 1
    return correct


def other_metric(group):
    evaluations = [
        [0, 1, 0],
        [0.75, 0.75, -0.5],
        [2, 0, -1],
        [2, -0.5, -0.5],
        [1, -1, 1],
        [-0.5, -0.5, 2],
        [-1, 0, 2],
        [-0.5, 0.75, 0.75],
    ]
    dists = 0
    for idx in range(8):
        e = evaluations[idx]
        g = group[idx]
        dist = sum([g[i] * e[i] for i in range(len(g))])
        dists += dist
    return dists


def display_weights():
    model = snake.brain.model
    weights = model.layers[0].get_weights()[0]
    biases = model.layers[0].get_weights()[1]
    body, apple, wall = [weights[i*8:(i+1)*8] for i in range(3)]
    b_1 = get_strength(body)
    a_1 = get_strength(apple)
    w_1 = get_strength(wall)
    b_2 = get_closeness(body)
    a_2 = get_closeness(apple)
    w_2 = get_closeness(wall)
    b_3 = other_metric(body)
    a_3 = other_metric(apple)
    w_3 = other_metric(wall)
    print("Strengths:", a_1, b_1, w_1)
    print("Gucciness:", a_1 - b_1 - w_1)
    print("Correct thinking:", a_2, b_2, w_2)
    print("Gucciness:", a_2 + (8 - b_2) + (8 - w_2))
    print("Other metric:", a_3, b_3, w_3)
    print("Gucciness:", a_3 - b_3 -  w_3)


snake = AI()
history = "game/training/first/"

for i in range(100):
    print()
    display = NoDisplay()
    validate = False
#    if i % 1 == -1:
#        print("Displaying: ", i)
#        display = Display()
#        validate = True
    env = Environment(snake, display)
    env.run(validate)
    env.save(history + str(i) + ".txt")
    display_weights()

print("Final epsilon:", snake.epsilon)


#snake.save_model('ganggang.h5')