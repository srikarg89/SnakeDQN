#https://towardsdatascience.com/deep-reinforcement-learning-build-a-deep-q-network-dqn-to-play-cartpole-with-tensorflow-2-and-gym-8e105744b998
from players.ai_single import AI
from game.env import Environment
from game.display import Display, NoDisplay

snake = AI()

gottem = 0
for i in range(10000):
    display = NoDisplay()
#    if i % 500 == 0:
#        print("Displaying: ", i)
#        display = Display()
    env = Environment(snake, display)
    env.run()
#    if len(env.snake.body) != 3:
#        gottem += 1

print("Gottem: ", gottem)
print("Final epsilon:", snake.epsilon)

# serialize model to JSON
#model_json = snake.train_brain.model.to_json()
#with open("models/first.json", "w") as json_file:
#    json_file.write(model_json)
# serialize weights to HDF5
#snake.train_brain.model.save_weights("models/first.h5")
#print("Saved model to disk")