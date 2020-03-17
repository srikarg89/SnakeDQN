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
        self.brain = Brain(Space.STATE_SIZE, Space.ACTION_SIZE)

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


    def save(self, state, action, ate, next_state):
#        print("SAVING: ", state.shape)
        reward = -1 if next_state is None else 1 if ate else 0
        self.brain.add_experience([state, action, reward, next_state])
        loss = self.brain.replay()


    def terminate(self, state, action, validate):
        self.running_length += len(self.body) + 1
        self.save(state, action, False, None)
        loss = self.brain.replay()
        self.running_loss += loss

        # Log stuff
        game_reward = len(self.body) + 1 - constants.SNAKE_INIT_LENGTH - 20
        self.rewards.append(game_reward)
        display_interval = 50.0
        if len(self.rewards) >= display_interval:
            self.rewards.popleft()
        if self.game_num % 50 == 0:
            print("Game: {}, Length: {}".format(self.game_num, self.running_length / display_interval))
            print("Epsilon: {}".format(self.epsilon))
            print("Loss: {}".format(self.running_loss / display_interval))
            self.running_length = 0.0
            self.running_loss = 0.0
        if validate:
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
        self.loss = tf.keras.losses.MeanSquaredError()
        self.model = Model(num_inputs, constants.HIDDEN, num_outputs)
        self.experiences = deque([])
        self.max_experiences = constants.MAX_EXPERIENCES
        self.min_experiences = constants.MIN_EXPERIENCES
        self.batch_size = constants.BATCH_SIZE
        self.model.compile(self.optimizer, self.loss)


    def predict(self, input):
        assert(type(input) == np.ndarray)
        reshaped = np.array([input])
        return self.model.predict(reshaped)


    def replay(self):
        if len(self.experiences) < self.min_experiences:
            return -100000
        # Sample from experience
        idxs = np.random.randint(low=0, high=len(self.experiences), size=self.batch_size)
        batch = [self.experiences[i] for i in idxs]
        states = np.array([exp[0] for exp in batch])
        next_states = np.array([exp[3] if exp[3] is not None else np.zeros((self.num_inputs, )) for exp in batch])
        q_s_a = self.model.predict_on_batch(states)
        q_s_a_d = self.model.predict_on_batch(next_states)
        # Setup training arrays
        x = np.zeros((len(batch), self.num_inputs))
        y = np.zeros((len(batch), self.num_outputs))
        for i, b in enumerate(batch):
            state, action, reward, next_state = b[0], b[1], b[2], b[3]
            # get the current q values for all actions in state
            current_q = q_s_a[i].numpy()
            # update the q value for action
            if next_state is None:
                # in this case, the game completed after action, so there is no max Q(s',a')
                # prediction possible
                current_q[action] = reward
            else:
                current_q[action] = reward + self.gamma * np.amax(q_s_a_d[i])
            x[i] = state
            y[i] = current_q

        output = self.model.train_on_batch(x, y)
        return output


    def get_action(self, state):
        pred = self.predict(state)
        return np.argmax(pred[0])
    

    def add_experience(self, exp):
        self.experiences.append(exp)
        if len(self.experiences) > self.max_experiences:
            self.experiences.popleft()
    

