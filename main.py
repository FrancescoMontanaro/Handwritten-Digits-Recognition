import tensorflow as tf

# Data Loading
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print(len(x_train))