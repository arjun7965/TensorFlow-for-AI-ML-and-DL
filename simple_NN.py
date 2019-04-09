import tensorflow as tf
from tensorflow import keras
import numpy as np

#create a simple neural network
#It has 1 layer, and that layer has 1 neuron, and the input shape to it is just 1 value.
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

#y=2x - 1. We are trying to make a model to predict y for a given x value
#compile. We use 'MEAN SQUARED ERROR' for the loss and 'STOCHASTIC GRADIENT DESCENT' for the optimizer
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0, 2.0, 5.0, -2.0, 6.0, 3.0, 8.0], dtype=float)
ys = np.array([-3.0, 3.0, 9.0, -5.0, 11.0, 5.0, 15.0], dtype=float)

#train the Neural Network. it trains for the number of epochs you specify
model.fit(xs, ys, epochs=500)

#predict the value of y for a given x
print(model.predict([10.0]))
