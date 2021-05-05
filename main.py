import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd


# Global Variables
bs = 8
num_classes = 10
img_h = 28
img_w = 28
apply_data_augmentation = True

SEED = 1234
tf.random.set_seed(SEED)

# Data Loading
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# Data augmentation
if apply_data_augmentation:
  train_data_gen = ImageDataGenerator(width_shift_range=10, height_shift_range=10, zoom_range=0.1, horizontal_flip=True, fill_mode='nearest', rescale=1./255)
else:
  train_data_gen = ImageDataGenerator(rescale=1./255)

valid_data_gen = ImageDataGenerator(rescale=1./255)
test_data_gen = ImageDataGenerator(rescale=1./255)