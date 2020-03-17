import tensorflow as tf
import numpy as np

arr1 = np.random.randint(4, size=(64,))
arr2 = tf.one_hot(arr1, 3)
print(arr2.shape)