import random
import datetime
import numpy as np
import tensorflow as tf
from game.helpers import opp
from collections import deque
from players.models.basic import Model
from players.snake import Snake
import game.constants as constants
from players.spaces import FPVSpace as Space

class AI(Snake):

    def __init__(self):
        super().__init__()
        self.directions = [constants.NORTH, constants.SOUTH, constants.EAST, constants.SOUTH]

        # DQNs
        print("Creating nets")
        self.train_brain = Brain(Space.STATE_SIZE, Space.ACTION_SIZE)
        print("Created training net")
        self.target_brain = Brain(Space.STATE_SIZE, Space.ACTION_SIZE)
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
        self.running_length = 0.0


    def get_state(self, env):
        self.state = Space.get_state(self, env)
        return self.state


    def act(self, env):
        self.action = Space.get_action(self, env)
        return Space.interpret(self, self.action)


    def train(self, exp):
        self.train_brain.add_experience(exp)
        self.train_brain.train(self.target_brain)
        if self.counter % constants.COPY_STEP == 0:
            self.target_brain.copy_weights(self.train_brain)


    def save(self, env):
        state = np.copy(self.state)
        action = self.action
#        print("SAVING: ", state.shape)
        reward = 1 if self.ate else 0
        new_state = self.get_state(env)
        done = False
        self.train([state, action, reward, new_state, done])


    def terminate(self, env):
        self.running_length += len(self.body) + 1
        # Save experience
        state = np.copy(self.state)
        action = self.action
        reward = -20
        next_state = self.get_state(env)
        done = True
        self.train([state, action, reward, next_state, done])

        # Log stuff
        game_reward = len(self.body) + 1 - constants.SNAKE_INIT_LENGTH - 20
        self.rewards.append(game_reward)
        if len(self.rewards) == 100:
            self.rewards.popleft()
        if self.game_num % 100 == 0:
            print("Game: {}, Length: {}".format(self.game_num, self.running_length / 100.0))
            self.running_length = 0.0
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
        #FIXME: wtf it doesn't care what the input shape is? nani?
        print("Predicting:", input.shape)
        return self.model(np.atleast_2d(input.astype('float32')))


#    @tf.function
    def train(self, target):
        if len(self.experiences) < self.min_experiences:
            return
        print("Training")

        idxs = np.random.randint(low=0, high=len(self.experiences), size=self.batch_size)
        experiences = [self.experiences[i] for i in idxs]
        temp = []
        for i in range(5):
            temp.append(np.asarray([exp[i] for exp in experiences]))
        states, actions, rewards, next_states, dones = temp
        next_values = np.max(target.predict(next_states), axis=1)
        actual_values = np.where(dones, rewards, rewards + self.gamma*next_values)

        with tf.GradientTape() as tape:
            print(type(states[0]), states[0].shape, states.shape)
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
            v1.assign(v2.numpy())



