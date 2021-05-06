import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import math
import sys


# Global Variables and Initializations
bs = 32
num_classes = 10
img_h = 28
img_w = 28
split_percentage = 0.9
apply_data_augmentation = True

SEED = 1234
tf.random.set_seed(SEED)


# Loads the MNIST dataset into arrays and performs some
# preprocessing in order to prepare it for the training.
def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.reshape((x_train.shape[0], img_h, img_w, 1))
    x_test = x_test.reshape((x_test.shape[0], img_h, img_w, 1))

    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)


# Function to create the Model
def create_model():
    # Hyperparameters
    start_filters = 8
    depth = 4
    lr = 1e-4

    model = tf.keras.Sequential()

    # Convolutional Layers
    for i in range(depth):
        if i == 0:
            input_shape = [img_h, img_w, 1]
        else:
            input_shape=[None]

        model.add(tf.keras.layers.Conv2D(
            filters=start_filters, 
            kernel_size=(3, 3), 
            strides=(1, 1), 
            padding='same', 
            input_shape=input_shape))

        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.MaxPool2D(pool_size = (2, 2)))

        start_filters *= 2

    # Fully Connected Layers
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(.3))
    model.add(tf.keras.layers.Dense(units = 256, activation = 'relu'))
    model.add(tf.keras.layers.Dropout(.3))
    model.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))
    model.add(tf.keras.layers.Dropout(.3))
    model.add(tf.keras.layers.Dense(units = 64, activation = 'relu'))
    model.add(tf.keras.layers.Dense(units = num_classes, activation = 'softmax'))

    # Loss functions
    loss = tf.keras.losses.CategoricalCrossentropy()

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate = lr)

    # Metrics
    metrics = ['accuracy']

    # Compile Model
    model.compile(optimizer = optimizer, loss = loss, metrics = metrics)

    model.summary()

    return model


# Function to evaluate the model's performances
# For each image of the test set it predicts 
# its class and compares it with its true label.
def evaluate_model(model, x_test, y_test):
    correctly_predicted = 0

    for i in range(len(x_test)):
        test_image = np.array(x_test[i])
        test_image = np.expand_dims(test_image, 0)

        prediction = model.predict(test_image)

        prediction = np.argmax(prediction)
        true_label = np.argmax(y_test[i])

        if(prediction == true_label):
            correctly_predicted += 1

        sys.stdout.write("\rImages evaluated %s / %s" % (str((i + 1)), str(len(x_test))))
        sys.stdout.flush()

    return correctly_predicted / len(x_test)


# Data Loading
(x_train, y_train), (x_test, y_test) = load_data()


# Data Splitting in training and validation set
# according to the splitting percentage defined above
split_index = math.floor(split_percentage * len(x_train))

(train_data, train_labels) = (x_train[:split_index], y_train[:split_index])
(valid_data, valid_labels) = (x_train[split_index:], y_train[split_index:])


# Data Augmentation
if apply_data_augmentation:
    train_data_gen = ImageDataGenerator(
        rotation_range = 10,
        width_shift_range = 10,
        height_shift_range = 10,
        zoom_range = 0.1,
        horizontal_flip = True,
        rescale = 1./255)
else:
    train_data_gen = ImageDataGenerator(rescale = 1./255)

valid_data_gen = ImageDataGenerator(rescale = 1./255)


# Data generators
train_gen = train_data_gen.flow(
    train_data, 
    train_labels, 
    batch_size = bs, 
    shuffle = True, 
    seed = SEED)

valid_gen = valid_data_gen.flow(
    valid_data, 
    valid_labels, 
    batch_size = bs, 
    shuffle = False, 
    seed = SEED)


# Datasets creation
train_dataset = tf.data.Dataset.from_generator(
    lambda: train_gen, 
    output_types=(tf.float32, tf.float32), 
    output_shapes=([None, img_h, img_w, 1], [None, num_classes]))

valid_dataset = tf.data.Dataset.from_generator(
    lambda: valid_gen, 
    output_types=(tf.float32, tf.float32), 
    output_shapes=([None, img_h, img_w, 1], [None, num_classes]))

# Repeating the datasets in order to use them across the
# different training epochs
train_dataset = train_dataset.repeat()
valid_dataset = valid_dataset.repeat()


# Model definition
model = create_model()


# Early Stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)


# Model training
sys.stdout.write("\nStarting the training...\n")

STEP_SIZE_TRAIN = train_gen.n / train_gen.batch_size
STEP_SIZE_VALID = valid_gen.n / valid_gen.batch_size

model.fit(
    x = train_dataset,
    epochs = 30,
    steps_per_epoch = STEP_SIZE_TRAIN,
    validation_data = valid_dataset,
    validation_steps = STEP_SIZE_VALID,
    callbacks = [early_stopping])

sys.stdout.write("\nTraining Completed\n")


# Saving the Model
sys.stdout.write("\nSaving the Model...\n")
model.save('saved_model')
sys.stdout.write("\nModel Saved\n")


# Model evaluation
sys.stdout.write("\nStarting Model Evaluation...\n")

final_accuracy = evaluate_model(model, x_test, y_test)

sys.stdout.write("\nModel's Accuracy on the Test set: %s" % final_accuracy)