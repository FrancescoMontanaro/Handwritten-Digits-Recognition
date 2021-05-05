import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import math

from tensorflow.python.ops.array_ops import repeat


# Global Variables and Initializations
bs = 32
num_classes = 10
img_h = 28
img_w = 28
split_percentage = 0.9
apply_data_augmentation = True

SEED = 1234
tf.random.set_seed(SEED)


def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.reshape((x_train.shape[0], img_h, img_w, 1))
    x_test = x_test.reshape((x_test.shape[0], img_h, img_w, 1))

    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)


# Data Loading
(x_train, y_train), (x_test, y_test) = load_data()


# Data Splitting
split_index = math.floor(split_percentage * len(x_train))

(train_data, train_labels) = (x_train[:split_index], y_train[:split_index])
(valid_data, valid_labels) = (x_train[split_index:], y_train[split_index:])


# Data Augmentation
if(apply_data_augmentation):
    train_data_gen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=10,
        height_shift_range=10,
        zoom_range=0.1,
        horizontal_flip=True,
        rescale=1./255)
else:
    train_data_gen = ImageDataGenerator(rescale=1./255)

valid_data_gen = ImageDataGenerator(rescale=1./255)


# Data generators
train_gen = train_data_gen.flow(train_data, train_labels, batch_size=bs, shuffle=True, seed=SEED)
valid_gen = valid_data_gen.flow(valid_data, valid_labels, batch_size=bs, shuffle=False, seed=SEED)


# Datasets creation
train_dataset = tf.data.Dataset.from_generator(lambda: train_gen, output_types=(tf.float32, tf.float32), output_shapes=([None, img_h, img_w, 1], [None, num_classes]))
valid_dataset = tf.data.Dataset.from_generator(lambda: valid_gen, output_types=(tf.float32, tf.float32), output_shapes=([None, img_h, img_w, 1], [None, num_classes]))

train_dataset.repeat()
valid_dataset.repeat()

print(train_dataset)
print(valid_dataset)