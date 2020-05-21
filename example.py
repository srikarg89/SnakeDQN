import numpy as np
#from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import ray
import tensorflow.compat.v1 as tf
import time

@ray.remote
class TrainingActor(object):
    def __init__(self, seed):
        print('Set new seed:', seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)
#        self.mnist = tfds.load(name="mnist", split=tfds.Split.TRAIN)
        self.mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

        # Setting up the softmax architecture.
        self.x = tf.placeholder('float', [None, 784])
        W = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))
        self.y = tf.nn.softmax(tf.matmul(self.x, W) + b)

        # Setting up the cost function.
        self.y_ = tf.placeholder('float', [None, 10])
        cross_entropy = -tf.reduce_sum(self.y_*tf.log(self.y))
        self.train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

        # Initialization
        self.init = tf.initialize_all_variables()
        self.sess = tf.Session(
            config=tf.ConfigProto(
                inter_op_parallelism_threads=1,
                intra_op_parallelism_threads=1
            )
        )

    def train(self):
        self.sess.run(self.init)

        for i in range(1000):
            batch_xs, batch_ys = self.mnist.train.next_batch(100)
            self.sess.run(self.train_step, feed_dict={self.x: batch_xs, self.y_: batch_ys})

        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

        return self.sess.run(accuracy, feed_dict={self.x: self.mnist.test.images,
                                                  self.y_: self.mnist.test.labels})


if __name__ == '__main__':
    # Start Ray.
    ray.init()

    # Create 3 actors.
    training_actors = [TrainingActor.remote(seed) for seed in range(3)]

    # Make them all train in parallel.
    accuracy_ids = [actor.train.remote() for actor in training_actors]
    print(ray.get(accuracy_ids))

    # Start new training runs in parallel.
    accuracy_ids = [actor.train.remote() for actor in training_actors]
    print(ray.get(accuracy_ids))