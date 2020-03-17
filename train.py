#https://towardsdatascience.com/deep-reinforcement-learning-build-a-deep-q-network-dqn-to-play-cartpole-with-tensorflow-2-and-gym-8e105744b998
from players.ai_single import AI
from game.env import Environment
from game.display import Display, NoDisplay

snake = AI()

for i in range(100):
    display = NoDisplay()
    validate = False
    if i % 50 == 0:
        print("Displaying: ", i)
        display = Display()
        validate = True
    env = Environment(snake, display)
    env.run(validate)
#    print(1/0)

print("Final epsilon:", snake.epsilon)

snake.save_model('first.h5')