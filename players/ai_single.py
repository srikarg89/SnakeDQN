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
from tensorflow.keras import backend as K

#tf.debugging.set_log_device_placement(True)
#NUM_THREADS = 4
#sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
#  intra_op_parallelism_threads=NUM_THREADS))

#K.set_session(sess)
K.set_floatx('float64')

class AI(Snake):

    def __init__(self, filename=None):
        super().__init__()
        self.directions = [constants.NORTH, constants.SOUTH, constants.EAST, constants.SOUTH]

        # DQNs
        print("Creating nets")
        self.brain = Brain(Space.STATE_SIZE, Space.ACTION_SIZE, filename)

        # Hyperparameters
        self.epsilon = 1.00
        self.min_epsilon = constants.MIN_EPSILON
        self.espilon_decay = constants.EPSILON_DECAY

        # Saving
        self.counter = 0
        self.rewards = deque([])
        self.game_num = 0

        # Logging
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = 'logs/dqn/' + current_time
        self.summary_writer = tf.summary.create_file_writer(log_dir)
        self.running_length = 0.0
        self.running_loss = 0.0


    def get_state(self, env):
        return Space.get_state(self, env)


    def act(self, state, validate):
        if np.random.random() < self.epsilon and not validate:
            action = np.random.choice(Space.ACTION_SIZE)
        else:
            action = Space.get_action(self, state)
        return action, Space.interpret(self, action)


    def remember(self, state, action, ate, next_state):
        reward = constants.DEATH_REWARD if next_state is None else constants.EAT_REWARD if ate else constants.LIFE_REWARD
        self.brain.add_experience([state, action, reward, next_state])
    

    def replay(self):
        loss = self.brain.replay()
        self.running_loss += loss
        self.epsilon = max(self.epsilon * self.espilon_decay, self.min_epsilon)


    def save_model(self, filename):
        self.brain.model.save('models/' + filename)


    def terminate(self, state, action, validate):
        self.running_length += len(self.body) + 1
        if not validate:
            self.remember(state, action, False, None)
            self.replay()

        # Log stuff
        game_reward = len(self.body) + 1 - constants.SNAKE_INIT_LENGTH - 20
        self.rewards.append(game_reward)
        display_interval = 10.0
        if len(self.rewards) >= display_interval:
            self.rewards.popleft()
        if self.game_num % display_interval == 0:
            print("Game: {}, Length: {}".format(self.game_num, self.running_length / display_interval))
            print("Epsilon: {}".format(self.epsilon))
            print("Loss: {}".format(self.running_loss / display_interval))
            self.running_length = 0.0
            self.running_loss = 0.0
        toprint = "Validation " if validate else ""
        print(toprint + "Game: {}, Length: {}".format(self.game_num, len(self.body) + 1))
#        self.epsilon = max(self.epsilon * self.espilon_decay, self.min_epsilon)
        with self.summary_writer.as_default():
            tf.summary.scalar('episode reward', game_reward, step=self.game_num)
            tf.summary.scalar('running avg reward(100)', sum(self.rewards) / len(self.rewards), step=self.game_num)
        self.game_num += 1


class Brain:

    def __init__(self, num_inputs, num_outputs, filename):
        # Model generation
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.learning_rate = constants.LEARNING_RATE
        self.gamma = constants.GAMMA
        self.optimizer = tf.optimizers.Adam(self.learning_rate)
        self.loss = tf.keras.losses.MeanSquaredError()
        self.model = Model(num_inputs, constants.HIDDEN, num_outputs, filename)
        self.max_experiences = constants.MAX_EXPERIENCES
        self.min_experiences = constants.MIN_EXPERIENCES
        self.experiences = deque([], maxlen=self.max_experiences)
        self.batch_size = constants.BATCH_SIZE
        self.model.compile(self.optimizer, self.loss)
        self.model._experimental_run_tf_function = True

    def predict(self, input):
        assert(type(input) == np.ndarray)
        return self.model(input).numpy()


    def replay(self):
        if len(self.experiences) < self.min_experiences:
            return -100000

        batch = random.sample(self.experiences, self.batch_size)
        states = np.array([b[0] for b in batch]).reshape(self.batch_size, self.num_inputs)
        next_states = np.array([b[3] if b[3] is not None else b[0] for b in batch]).reshape(self.batch_size, self.num_inputs)
        current_qs = self.model(states)
        next_qs = self.model(next_states)
        pred_qs = []
        loss = 0
        for i, b in enumerate(batch):
            state, action, reward, next_state = b[0], b[1], b[2], b[3]
            current_q = current_qs[i].numpy()
#            current_q = self.predict(state)
            # update the q value for action
            if next_state is None:
                current_q[action] = reward
            else:
                next_q = next_qs[i].numpy()
#                next_q = self.predict(next_state)
                current_q[action] = reward + self.gamma * np.amax(next_q)

            pred_qs.append(current_q)

        history = self.model.fit(states, np.array(pred_qs), verbose=False)
        loss += history.history['loss'][0]

        return loss / self.batch_size


    def get_action(self, state):
        pred = self.predict(state)
        return np.argmax(pred[0])
    

    def add_experience(self, exp):
        self.experiences.append(exp)
   

