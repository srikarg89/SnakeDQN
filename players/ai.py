import random
import datetime
import numpy as np
import tensorflow as tf
from game.helpers import opp
from collections import deque
from players.snake import Snake
import game.constants as constants

class AI(Snake):

    def __init__(self):
        super().__init__()
        self.board_width, self.board_height = constants.BOARD_WIDTH, constants.BOARD_HEIGHT
        self.directions = [constants.NORTH, constants.SOUTH, constants.EAST, constants.SOUTH]

        # DQNs
        print("Creating nets")
        self.train_brain = Brain(800, 4)
        print("Created training net")
        self.target_brain = Brain(800, 4)
        print("Created target net")

        # Hyperparameters
        self.epsilon = 1.00
        self.min_epsilon = constants.MIN_EPSILON
        self.espilon_decay = constants.EPSILON_DECAY

        # Saving
        self.state = None
        self.action = None
        self.counter = 0
        self.rewards = deque([])
        self.game_num = 0

        # Logging
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = 'logs/dqn/' + current_time
        self.summary_writer = tf.summary.create_file_writer(log_dir)


    def get_state(self, env):
        arr1, arr2 = [], []
        for i in range(self.board_height):
            for j in range(self.board_width):
                a = 1 if (i,j) == env.apple else 0
                s = 1 if (i,j) in [self.head] + self.body else 0
                arr1.append(s)
                arr2.append(a)
        return np.array(arr1 + arr2)


    def act(self, env):
        # Calculate stuff
        state = self.get_state(env)
        idx = self.train_brain.get_action(state, self.epsilon)
        # Save variables to add to experience replay later
        self.state = state
        self.action = idx

        return self.directions[idx]
    

    def learn(self, exp):
        self.train_brain.add_experience(exp)
        self.train_brain.learn(self.target_brain)
        if self.counter % constants.COPY_STEP == 0:
            self.target_brain.copy_weights(self.train_brain)


    def save(self, env):
        state, action = self.state, self.action
        reward = 1 if self.ate else 0
        new_state = self.get_state(env)
        done = False
        self.learn([state, action, reward, new_state, done])


    def terminate(self, env):
        # Save experience
        state, action = self.state, self.action
        reward = -20
        next_state = self.get_state(env)
        done = True
        self.learn([state, action, reward, next_state, done])

        # Log stuff
        game_reward = len(self.body) + 1 - constants.SNAKE_INIT_LENGTH - 20
        self.rewards.append(game_reward)
        if len(self.rewards) == 100:
            self.rewards.popleft()
        if self.game_num % 100 == 0:
            print("Game: {}, Length: {}".format(self.game_num, len(self.body) + 1))
        self.epsilon = max(self.epsilon * self.espilon_decay, self.min_epsilon)
        with self.summary_writer.as_default():
            tf.summary.scalar('episode reward', game_reward, step=self.game_num)
            tf.summary.scalar('running avg reward(100)', sum(self.rewards) / len(self.rewards), step=self.game_num)
        self.game_num += 1


class Brain:

    def __init__(self, num_inputs, num_outputs):
        # Model generation
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.learning_rate = constants.LEARNING_RATE
        self.gamma = constants.GAMMA
        self.optimizer = tf.optimizers.Adam(self.learning_rate)
        self.model = Model(num_inputs, constants.HIDDEN, num_outputs)
        self.experiences = deque([])
        self.max_experiences = constants.MAX_EXPERIENCES
        self.min_experiences = constants.MIN_EXPERIENCES
        self.batch_size = constants.BATCH_SIZE


    def predict(self, input):
        return self.model(np.atleast_2d(input.astype('float32')))


    @tf.function
    def learn(self, target):
        if len(self.experiences) < self.min_experiences:
            return
        
        idxs = np.random.randint(low=0, high=len(self.experiences), size=self.batch_size)
        experiences = [self.experiences[i] for i in idxs]
        temp = []
        for i in range(5):
            temp.append(np.asarray([experiences[j][i] for j in experiences]))
        states, actions, rewards, next_states, dones = temp
        next_values = np.max(target.predict(next_states), axis=1)
        actual_values = np.where(dones, rewards, rewards + self.gamma*next_values)

        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(
                self.predict(states) * tf.one_hot(actions, self.num_outputs), axis=1)
            loss = tf.math.reduce_sum(tf.square(actual_values - selected_action_values))
            variables = self.model.trainable_variables
            gradients = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))


    def get_action(self, state, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.num_outputs)
        return np.argmax(self.predict(np.atleast_2d(state))[0])
    

    def add_experience(self, exp):
        self.experiences.append(exp)
        if len(self.experiences) > self.max_experiences:
            self.experiences.popleft()
    

    def copy_weights(self, train):
        variables1 = self.model.trainable_variables
        variables2 = train.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assing(v2.numpy())



class Model(tf.keras.Model):

    def __init__(self, num_inputs, hidden_units, num_outputs):
        super(Model, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape = (num_inputs, ))
        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(
                i, activation='tanh', kernel_initializer='RandomNormal'
            ))
        self.output_layer = tf.keras.layers.Dense(
            num_outputs, activation='linear', kernel_initializer='RandomNormal'
        )

    @tf.function
    def call(self, inputs):
        output = self.input_layer(inputs)
        for layer in self.hidden_layers:
            output = layer(output)
        output = self.output_layer(output)
        return output
        

