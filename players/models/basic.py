import tensorflow as tf


def Model(num_inputs, hidden_units, num_outputs, filename):
    return build(num_inputs, hidden_units, num_outputs)

def build(num_inputs, hidden_units, num_outputs):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape = (num_inputs, )))
    for i in hidden_units:
        model.add(tf.keras.layers.Dense(
            i, activation='tanh', kernel_initializer='RandomNormal'
        ))
    model.add(tf.keras.layers.Dense(
        num_outputs, activation='linear', kernel_initializer='RandomNormal'
    ))
    model.summary()
    return model


def load(filename):
    return tf.keras.models.load_model(filename)


"""
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
        y = self.input_layer(inputs)
        for layer in self.hidden_layers:
            y = layer(y)
        y = self.output_layer(y)
        return y
"""

#model = Model(1200, [400, 400], 4)
#print(model.trainable_variables)