from players.ai import AI
from game.env import Environment
from game.display import NoDisplay

snake = AI()

for i in range(1000):
    env = Environment(snake, NoDisplay())
    env.run()


# serialize model to JSON
model_json = snake.train_brain.model.to_json()
with open("models/first.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
snake.train_brain.model.save_weights("models/first.h5")
print("Saved model to disk")