import tensorflow as tf
import numpy as np
from tensorflow import keras


model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
xs = np.array([1.0, 3.0, 6.0, 10.0, 2.0, 9.0, 12.0], dtype=float)
ys = np.array([1.0, 2.0, 3.5, 5.5, 1.5, 5.0, 6.5], dtype=float)

#train the NN for 500 epochs
model.fit(xs, ys, epochs=500)

#predict the value of the house for 7 bedrooms
print(model.predict([7.0]))
